"""Canonical five-task full-fine-tuning pipeline.

Every task/replicate starts independently from the same pretrained LaBraM
backbone W0 and trains for the fixed 20-epoch endpoint.  This module contains no
spectral analysis and no plotting.
"""

from __future__ import annotations

import gc
import math
import os
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np

from .config import EXPECTED_MODULE_SHAPES, TASK_ORDER, ProjectConfig
from .data import (
    TaskMeta,
    audit_h5_data,
    audit_subject_composition,
    class_weights,
    load_all_task_meta,
    load_probe_batch,
    load_task_into_memory,
    make_loaders,
    preflight_probe_indices,
    sha256_ndarray,
)
from .io import (
    audit_delta_archives,
    canonical_json_sha256,
    discover_runs,
    dump_json,
    ensure_dir,
    load_json,
    log,
    read_csv,
    run_dir,
    sanitize_module_key,
    sha256_file,
    stable_seed,
    write_csv,
)
from .model import (
    build_labram_backbone,
    create_task_classifier,
    evaluate_classifier,
    extract_tsv_weights,
    require_torch,
)


EPS = 1e-12


def seed_everything(seed: int) -> None:
    torch, _, _ = require_torch()
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def make_optimizer(classifier, project: ProjectConfig):
    torch, _, _ = require_torch()
    cfg = project.training
    return torch.optim.AdamW(
        [
            {
                "params": [p for p in classifier.backbone.parameters() if p.requires_grad],
                "lr": float(cfg.backbone_lr),
                "weight_decay": float(cfg.weight_decay),
            },
            {
                "params": [p for p in classifier.head.parameters() if p.requires_grad],
                "lr": float(cfg.head_lr),
                "weight_decay": 0.0,
            },
        ],
        betas=(float(cfg.beta1), float(cfg.beta2)),
        eps=float(cfg.adam_eps),
    )


def train_one_epoch(classifier, loader, device, optimizer, weights) -> Dict[str, object]:
    torch, _, functional = require_torch()
    classifier.train()
    n_classes = int(classifier.head.out_features)
    confusion = torch.zeros((n_classes, n_classes), dtype=torch.int64, device=device)
    loss_numerator = torch.zeros((), dtype=torch.float64, device=device)
    loss_denominator = torch.zeros((), dtype=torch.float64, device=device)
    steps = 0
    for x, y in loader:
        x = x.to(device, non_blocking=False)
        y = y.to(device, non_blocking=False)
        optimizer.zero_grad(set_to_none=True)
        logits = classifier(x)
        loss = functional.cross_entropy(logits, y, weight=weights)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        # Protocol boundary: deliberately no gradient clipping.
        optimizer.step()
        with torch.no_grad():
            prediction = logits.argmax(dim=1)
            confusion += torch.bincount(
                y.to(torch.int64) * n_classes + prediction.to(torch.int64),
                minlength=n_classes * n_classes,
            ).reshape(n_classes, n_classes)
            denominator = weights[y].sum().to(torch.float64)
            loss_numerator += loss.detach().to(torch.float64) * denominator
            loss_denominator += denominator
        steps += 1

    from .model import metrics_from_confusion

    result = metrics_from_confusion(confusion.detach().cpu().numpy())
    result["objective_loss"] = float(
        (loss_numerator / torch.clamp(loss_denominator, min=EPS)).item()
    )
    result["optimizer_steps"] = int(steps)
    return result


def _atomic_torch_save(payload, path: Path) -> None:
    torch, _, _ = require_torch()
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _save_delta_archive(
    path: Path, base: Mapping[str, object], adapted: Mapping[str, object]
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    payload = {
        sanitize_module_key(module): (
            adapted[module] - base[module]
        ).numpy().astype(np.float32, copy=False)
        for module in sorted(base)
    }
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _protocol_payload(
    project: ProjectConfig,
    data_audits: Mapping[str, object],
    subject_audit: Mapping[str, object],
    base_digest: str,
) -> Dict[str, object]:
    cfg = project.training
    source_dir = Path(__file__).resolve().parent
    source_files = ("config.py", "data.py", "model.py", "training.py")
    return {
        "version": "clean-1.0",
        "tasks": list(TASK_ORDER),
        "replicates": cfg.replicates,
        "all_subjects_pooled": True,
        "session": 1,
        "window_seconds": 4,
        "sample_rate_hz": 200,
        "input_shape_before_model": [64, 4, 200],
        "input_chans": list(project.input_chans),
        "normalisation": "input already globally z-scored; no /100 and no second z-score",
        "full_fine_tuning": True,
        "epochs": cfg.epochs,
        "optimizer": {
            "name": "AdamW",
            "backbone_lr": cfg.backbone_lr,
            "head_lr": cfg.head_lr,
            "weight_decay": cfg.weight_decay,
            "betas": [cfg.beta1, cfg.beta2],
            "eps": cfg.adam_eps,
            "gradient_clipping": False,
            "scheduler": None,
            "warmup": None,
            "early_stopping": False,
            "amp": False,
        },
        "loss": "class-weighted cross entropy",
        "batch_size": cfg.batch_size,
        "eval_batch_size": cfg.eval_batch_size,
        "seed_base": cfg.seed_base,
        "model_init_seed": cfg.model_init_seed,
        "base_full_backbone_digest": base_digest,
        "checkpoint_sha256": sha256_file(project.checkpoint),
        "modeling_file_sha256": sha256_file(project.modeling_file),
        "implementation_source_sha256": {
            name: sha256_file(source_dir / name) for name in source_files
        },
        "data_audits": data_audits,
        "subject_composition": subject_audit,
        "analysed_matrices": {
            "count": len(EXPECTED_MODULE_SHAPES),
            "shapes": {key: list(value) for key, value in EXPECTED_MODULE_SHAPES.items()},
        },
    }


def _preflight_request_contract(
    project: ProjectConfig, metas: Mapping[str, TaskMeta]
) -> Dict[str, object]:
    """Cheap, code-bound identity checked before reusing a full preflight."""

    data_identity = {}
    for task in TASK_ORDER:
        meta = metas[task]
        stat = meta.path.stat()
        data_identity[task] = {
            "path": str(meta.path.resolve()),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "ctime_ns": int(stat.st_ctime_ns),
            "shape": list(meta.shape),
            "labels_sha256": sha256_ndarray(meta.labels),
            "subject_ids_sha256": sha256_ndarray(meta.subjects),
        }
    source_dir = Path(__file__).resolve().parent
    source_files = ("config.py", "data.py", "model.py", "training.py")
    return {
        "version": "clean-1.0",
        "training": asdict(project.training),
        "input_chans": list(project.input_chans),
        "device_type": str(project.device).split(":", 1)[0],
        "checkpoint": {
            "path": str(project.checkpoint.resolve()),
            "sha256": sha256_file(project.checkpoint),
        },
        "modeling_file": {
            "path": str(project.modeling_file.resolve()),
            "sha256": sha256_file(project.modeling_file),
        },
        "data_identity": data_identity,
        "preflight_source_sha256": {
            name: sha256_file(source_dir / name) for name in source_files
        },
    }


def run_preflight(project: ProjectConfig, force: bool = False) -> Dict[str, object]:
    project.validate(require_paths=True)
    ensure_dir(project.root)
    preflight_path = project.root / "preflight.json"
    success_path = project.root / "_PREFLIGHT_SUCCESS.json"
    metas = load_all_task_meta(project)
    request_contract = _preflight_request_contract(project, metas)
    request_signature = canonical_json_sha256(request_contract)
    if not force and preflight_path.is_file() and success_path.is_file():
        payload = load_json(preflight_path)
        marker = load_json(success_path)
        if (
            canonical_json_sha256(payload) == str(marker.get("preflight_sha256", ""))
            and str(payload.get("request_signature", "")) == request_signature
            and str(marker.get("request_signature", "")) == request_signature
        ):
            log("Preflight resume: retained matching certified audit")
            return payload

    data_audits = {}
    for task in TASK_ORDER:
        log(f"Preflight H5 audit: {task}")
        data_audits[task] = audit_h5_data(metas[task])
    subject_audit = audit_subject_composition(metas)

    torch, _, _ = require_torch()
    device = torch.device(project.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    backbone, checkpoint_audit, base_digest = build_labram_backbone(project)
    model_audits = {}
    for task in TASK_ORDER:
        indices = preflight_probe_indices(metas[task], n=32)
        x, _ = load_probe_batch(metas[task], indices)
        classifier = create_task_classifier(
            backbone,
            metas[task].n_classes,
            project.input_chans,
            stable_seed(project.training.seed_base, task, 1, "head"),
        ).to(device)
        with torch.no_grad():
            logits = classifier(x.to(device))
        if tuple(logits.shape) != (len(indices), metas[task].n_classes):
            raise RuntimeError(f"{task}: model preflight output shape {tuple(logits.shape)}")
        model_audits[task] = {
            "probe_samples": len(indices),
            "logit_shape": list(logits.shape),
            "finite": bool(torch.isfinite(logits).all().item()),
        }
        if not model_audits[task]["finite"]:
            raise FloatingPointError(f"{task}: non-finite preflight logits")
        del classifier

    protocol = _protocol_payload(project, data_audits, subject_audit, base_digest)
    payload = {
        "status": "passed",
        "request_contract": request_contract,
        "request_signature": request_signature,
        "protocol": protocol,
        "protocol_fingerprint": canonical_json_sha256(protocol),
        "checkpoint_audit": checkpoint_audit,
        "model_audits": model_audits,
    }
    dump_json(preflight_path, payload)
    dump_json(
        success_path,
        {
            "status": "passed",
            "preflight_sha256": canonical_json_sha256(payload),
            "request_signature": request_signature,
            "protocol_fingerprint": payload["protocol_fingerprint"],
        },
    )
    del backbone
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log("Preflight passed")
    return payload


def _prepare_run_directory(path: Path, overwrite: bool) -> Path:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"Non-empty run directory exists: {path}; use overwrite only after checking it"
            )
        resolved = path.resolve()
        expected_parent = (path.parents[2] / "runs").resolve()
        if resolved.parent.parent != expected_parent:
            raise RuntimeError(f"Refusing to remove unexpected directory: {resolved}")
        shutil.rmtree(path)
    return ensure_dir(path)


def _run_is_complete(
    path: Path,
    task: str,
    replicate: int,
    protocol_fingerprint: str,
    expected_w0_digest: str,
) -> bool:
    required = [
        path / "metadata.json",
        path / "adapted_checkpoint.pth",
        path / "delta_weights.npz",
        path / "_SUCCESS.json",
        path / "epoch_metrics.csv",
    ]
    if not all(item.is_file() for item in required):
        return False
    try:
        metadata = load_json(path / "metadata.json")
        success = load_json(path / "_SUCCESS.json")
        epochs = read_csv(path / "epoch_metrics.csv")
        artifacts = dict(metadata.get("artifacts", {}))
        delta_hash = sha256_file(path / "delta_weights.npz")
        checkpoint_hash = sha256_file(path / "adapted_checkpoint.pth")
        return all(
            (
                str(metadata.get("task")) == task,
                int(metadata.get("replicate", -1)) == replicate,
                int(metadata.get("epochs", -1)) == 20,
                str(metadata.get("protocol_fingerprint", "")) == protocol_fingerprint,
                str(metadata.get("base_full_backbone_digest", "")) == expected_w0_digest,
                str(success.get("phase", "")) == "final",
                str(success.get("task", "")) == task,
                int(success.get("replicate", -1)) == replicate,
                str(success.get("protocol_fingerprint", "")) == protocol_fingerprint,
                str(success.get("base_full_backbone_digest", "")) == expected_w0_digest,
                len(epochs) == 20,
                sorted(int(float(row["epoch"])) for row in epochs) == list(range(1, 21)),
                str(artifacts.get("delta_weights_sha256", "")) == delta_hash,
                str(artifacts.get("adapted_checkpoint_sha256", "")) == checkpoint_hash,
                str(success.get("delta_weights_sha256", "")) == delta_hash,
                str(success.get("adapted_checkpoint_sha256", "")) == checkpoint_hash,
            )
        )
    except (KeyError, TypeError, ValueError, OSError):
        return False


def train_run(
    project: ProjectConfig,
    meta: TaskMeta,
    task_data,
    task: str,
    replicate: int,
    protocol_fingerprint: str,
    expected_w0_digest: str,
    overwrite: bool = False,
) -> Dict[str, object]:
    torch, _, _ = require_torch()
    device = torch.device(project.device)
    directory = run_dir(project.root, task, replicate)
    directory = _prepare_run_directory(directory, overwrite)
    cfg = project.training

    head_seed = stable_seed(cfg.seed_base, task, replicate, "head")
    runtime_seed = stable_seed(cfg.seed_base, task, replicate, "runtime", "final")
    loader_seed = stable_seed(cfg.seed_base, task, replicate, "loader", "final")
    backbone, checkpoint_audit, base_digest = build_labram_backbone(project)
    if base_digest != expected_w0_digest:
        raise RuntimeError(f"{task}/rep{replicate:02d}: W0 digest changed after preflight")
    base_tsv = extract_tsv_weights(backbone)
    classifier = create_task_classifier(
        backbone, meta.n_classes, project.input_chans, head_seed
    ).to(device)
    seed_everything(runtime_seed)
    train_loader, eval_loader = make_loaders(
        task_data, cfg.batch_size, cfg.eval_batch_size, loader_seed
    )
    weights = class_weights(meta, device)
    optimizer = make_optimizer(classifier, project)

    epoch_rows: List[Dict[str, object]] = []
    log(f"Training {task}/rep{replicate:02d}: {meta.n} samples, {cfg.epochs} epochs")
    for epoch in range(1, cfg.epochs + 1):
        started = time.time()
        metrics = train_one_epoch(classifier, train_loader, device, optimizer, weights)
        row = {
            "task": task,
            "replicate": replicate,
            "epoch": epoch,
            "n_samples": meta.n,
            "objective_loss": float(metrics["objective_loss"]),
            "accuracy": float(metrics["accuracy"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "chance_bacc": 1.0 / meta.n_classes,
            "optimizer_steps_this_epoch": int(metrics["optimizer_steps"]),
            "seconds": float(time.time() - started),
            "metric_scope": "online_train_pass",
        }
        if not all(
            math.isfinite(float(row[key]))
            for key in ("objective_loss", "accuracy", "balanced_accuracy", "macro_f1")
        ):
            raise FloatingPointError(f"{task}/rep{replicate:02d}: non-finite epoch metrics")
        epoch_rows.append(row)
        log(
            f"  epoch {epoch:02d}/{cfg.epochs}: loss={row['objective_loss']:.4f}, "
            f"bACC={row['balanced_accuracy']:.4f}, {row['seconds']:.1f}s"
        )
    write_csv(directory / "epoch_metrics.csv", epoch_rows)

    final_metrics = evaluate_classifier(
        classifier, eval_loader, device, meta.n_classes, weights
    )
    chance = 1.0 / meta.n_classes
    if float(final_metrics["balanced_accuracy"]) < chance + 0.05:
        raise RuntimeError(
            f"{task}/rep{replicate:02d}: final bACC is too close to chance"
        )
    adapted_tsv = extract_tsv_weights(backbone)
    delta_path = directory / "delta_weights.npz"
    _save_delta_archive(delta_path, base_tsv, adapted_tsv)
    checkpoint_path = directory / "adapted_checkpoint.pth"
    _atomic_torch_save(
        {
            "task": task,
            "replicate": replicate,
            "backbone": backbone.state_dict(),
            "head": classifier.head.state_dict(),
            "input_chans": list(project.input_chans),
            "protocol_fingerprint": protocol_fingerprint,
            "base_full_backbone_digest": base_digest,
        },
        checkpoint_path,
    )
    metadata = {
        "version": "clean-1.0",
        "task": task,
        "replicate": replicate,
        "num_classes": meta.n_classes,
        "raw_classes": [int(x) for x in meta.classes],
        "n_samples": meta.n,
        "n_subjects": int(len(np.unique(meta.subjects))),
        "all_subjects_pooled": True,
        "epochs": cfg.epochs,
        "protocol_fingerprint": protocol_fingerprint,
        "input_chans": list(project.input_chans),
        "head_seed": head_seed,
        "runtime_seed": runtime_seed,
        "loader_seed": loader_seed,
        "base_full_backbone_digest": base_digest,
        "checkpoint_audit": checkpoint_audit,
        "metrics_final": final_metrics,
        "optimizer": {
            "name": "AdamW",
            "backbone_lr": cfg.backbone_lr,
            "head_lr": cfg.head_lr,
            "betas": [cfg.beta1, cfg.beta2],
            "eps": cfg.adam_eps,
            "weight_decay": cfg.weight_decay,
            "gradient_clipping": False,
            "scheduler": None,
            "warmup": None,
            "early_stopping": False,
            "amp": False,
        },
        "loss": {
            "name": "class-weighted_cross_entropy",
            "weights": weights.detach().cpu().numpy().tolist(),
        },
        "ram_staging": {
            "nbytes": task_data.nbytes,
            "load_seconds": task_data.load_seconds,
        },
    }
    metadata["artifacts"] = {
        "delta_weights_sha256": sha256_file(delta_path),
        "adapted_checkpoint_sha256": sha256_file(checkpoint_path),
    }
    dump_json(directory / "metadata.json", metadata)
    dump_json(
        directory / "_SUCCESS.json",
        {
            "phase": "final",
            "task": task,
            "replicate": replicate,
            "protocol_fingerprint": protocol_fingerprint,
            "base_full_backbone_digest": base_digest,
            **metadata["artifacts"],
        },
    )
    result = {
        "task": task,
        "replicate": replicate,
        "n_samples": meta.n,
        "epochs": cfg.epochs,
        "final_bacc": float(final_metrics["balanced_accuracy"]),
        "protocol_fingerprint": protocol_fingerprint,
        "base_full_backbone_digest": base_digest,
    }
    del classifier, backbone, train_loader, eval_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def train_all(
    project: ProjectConfig,
    resume: bool = True,
    overwrite: bool = False,
    force_preflight: bool = False,
) -> Dict[str, object]:
    project.validate(require_paths=True)
    preflight = run_preflight(project, force=force_preflight)
    protocol = preflight["protocol"]
    protocol_fingerprint = str(preflight["protocol_fingerprint"])
    expected_w0 = str(protocol["base_full_backbone_digest"])
    metas = load_all_task_meta(project)
    rows = []
    for task in TASK_ORDER:
        task_data = load_task_into_memory(metas[task], project.training.ram_reserve_gb)
        try:
            for replicate in (1, 2):
                directory = run_dir(project.root, task, replicate)
                if resume and _run_is_complete(
                    directory,
                    task,
                    replicate,
                    protocol_fingerprint,
                    expected_w0,
                ):
                    metadata = load_json(directory / "metadata.json")
                    rows.append(
                        {
                            "task": task,
                            "replicate": replicate,
                            "n_samples": metadata["n_samples"],
                            "epochs": metadata["epochs"],
                            "final_bacc": metadata["metrics_final"]["balanced_accuracy"],
                            "protocol_fingerprint": metadata.get("protocol_fingerprint", ""),
                            "base_full_backbone_digest": metadata.get(
                                "base_full_backbone_digest", ""
                            ),
                        }
                    )
                    log(f"Training resume: kept {task}/rep{replicate:02d}")
                    continue
                rows.append(
                    train_run(
                        project,
                        metas[task],
                        task_data,
                        task,
                        replicate,
                        protocol_fingerprint,
                        expected_w0,
                        overwrite=overwrite,
                    )
                )
        finally:
            del task_data
            gc.collect()

    write_csv(project.root / "run_summary.csv", rows)
    runs = discover_runs(project.root)
    delta_audit = audit_delta_archives(runs)
    manifest = {
        "version": "clean-1.0",
        "protocol_fingerprint": protocol_fingerprint,
        "base_full_backbone_digest": expected_w0,
        "tasks": list(TASK_ORDER),
        "replicates": 2,
        "n_runs": len(runs),
        "runs": rows,
        "delta_archive_audit": delta_audit,
    }
    dump_json(project.root / "run_manifest.json", manifest)
    dump_json(
        project.root / "_FINAL_SUCCESS.json",
        {
            "status": "passed",
            "n_runs": len(runs),
            "protocol_fingerprint": protocol_fingerprint,
            "run_manifest_sha256": sha256_file(project.root / "run_manifest.json"),
        },
    )
    log("All ten full-fine-tuning runs completed and audited")
    return manifest
