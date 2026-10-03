"""LaBraM construction, checkpoint handling, and functional reconstruction.

Functional spectra load the complete adapted candidate checkpoint.  They then
replace only the 72 analysed matrices; the task head and every other adapted
backbone parameter remain untouched.  This is the central functional-retention
boundary of the experiment.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from .config import EXPECTED_MODULE_SHAPES, N_BLOCKS, ProjectConfig
from .data import TaskMeta, class_weights, load_task_into_memory, make_eval_loader
from .io import RunRecord, load_delta_weights


EPS = 1e-12
_IMPORTED_MODELING_FILES = set()


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for LaBraM training and functional evaluation"
        ) from exc
    return torch, nn, functional


def load_torch_file(path: Path):
    """Load trusted experiment artifacts across old and new PyTorch releases."""

    torch, _, _ = require_torch()
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        # ``weights_only`` is unavailable in older LaBraM environments.
        return torch.load(Path(path), map_location="cpu")


def import_modeling_file(path: Path) -> None:
    path = Path(path).expanduser().resolve()
    if path in _IMPORTED_MODELING_FILES:
        return
    if not path.is_file():
        raise FileNotFoundError(path)
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    name = f"labram_modeling_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import LaBraM modeling file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _IMPORTED_MODELING_FILES.add(path)


def torch_state_digest(state: Mapping[str, object]) -> str:
    torch, _, _ = require_torch()
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not torch.is_tensor(value):
            continue
        array = value.detach().cpu().contiguous().numpy()
        h.update(key.encode("utf-8"))
        h.update(str(array.dtype).encode("ascii"))
        h.update(str(tuple(array.shape)).encode("ascii"))
        h.update(array.tobytes(order="C"))
    return h.hexdigest()


_ALLOWED_CHECKPOINT_ONLY_PREFIXES = ("lm_head.", "projection_head.", "norm.")
_ALLOWED_CHECKPOINT_ONLY_EXACT = {
    "logit_scale",
    "mask_token",
    "head.weight",
    "head.bias",
}
_ALLOWED_MODEL_MISSING_PREFIXES = ("fc_norm.",)


def _allowed_checkpoint_only(key: str) -> bool:
    return key in _ALLOWED_CHECKPOINT_ONLY_EXACT or any(
        key.startswith(prefix) for prefix in _ALLOWED_CHECKPOINT_ONLY_PREFIXES
    )


def _allowed_model_missing(key: str) -> bool:
    return "relative_position_index" in key or any(
        key.startswith(prefix) for prefix in _ALLOWED_MODEL_MISSING_PREFIXES
    )


def strict_load_pretrained(model, checkpoint_path: Path) -> Dict[str, object]:
    torch, _, _ = require_torch()
    checkpoint = load_torch_file(checkpoint_path)
    state = checkpoint
    payload_name = "<root>"
    if isinstance(checkpoint, Mapping):
        for key in ("model", "module", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], Mapping):
                state = checkpoint[key]
                payload_name = key
                break
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint does not contain a state mapping")

    clean = {}
    for raw_key, value in state.items():
        key = str(raw_key)
        if key.startswith("student."):
            key = key[8:]
        if key.startswith("module."):
            key = key[7:]
        clean[key] = value

    model_state = model.state_dict()
    time_embed_transform = None
    if "time_embed" in clean and "time_embed" in model_state:
        if tuple(clean["time_embed"].shape) != tuple(model_state["time_embed"].shape):
            old_shape = tuple(clean["time_embed"].shape)
            target = int(model_state["time_embed"].shape[1])
            if clean["time_embed"].shape[0] != model_state["time_embed"].shape[0]:
                raise RuntimeError("time_embed batch dimension mismatch")
            if clean["time_embed"].shape[-1] != model_state["time_embed"].shape[-1]:
                raise RuntimeError("time_embed embedding dimension mismatch")
            if clean["time_embed"].shape[1] < target:
                raise RuntimeError("checkpoint time_embed is shorter than target")
            clean["time_embed"] = clean["time_embed"][:, :target]
            time_embed_transform = {
                "from": list(old_shape),
                "to": list(clean["time_embed"].shape),
            }

    checkpoint_only = sorted(key for key in clean if key not in model_state)
    unknown_checkpoint = [key for key in checkpoint_only if not _allowed_checkpoint_only(key)]
    if unknown_checkpoint:
        raise RuntimeError(f"Unknown checkpoint-only keys: {unknown_checkpoint[:20]}")
    mismatches = [
        (key, tuple(clean[key].shape), tuple(model_state[key].shape))
        for key in clean
        if key in model_state and tuple(clean[key].shape) != tuple(model_state[key].shape)
    ]
    if mismatches:
        raise RuntimeError(f"Checkpoint/model shape mismatches: {mismatches[:20]}")
    loadable = {
        key: value
        for key, value in clean.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    model_missing = sorted(key for key in model_state if key not in loadable)
    unknown_missing = [key for key in model_missing if not _allowed_model_missing(key)]
    if unknown_missing:
        raise RuntimeError(f"Unexpected model tensors absent from checkpoint: {unknown_missing[:20]}")
    missing, unexpected = model.load_state_dict(loadable, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected load_state_dict keys: {list(unexpected)[:20]}")
    bad_missing = [key for key in missing if not _allowed_model_missing(key)]
    if bad_missing:
        raise RuntimeError(f"Unexpected missing keys: {bad_missing[:20]}")
    return {
        "payload": payload_name,
        "checkpoint_tensor_count": len(clean),
        "model_tensor_count": len(model_state),
        "loaded_tensor_count": len(loadable),
        "checkpoint_only_allowed": checkpoint_only,
        "model_missing_allowed": model_missing,
        "time_embed_transform": time_embed_transform,
    }


def build_labram_backbone(project: ProjectConfig):
    torch, _, _ = require_torch()
    import_modeling_file(project.modeling_file)
    try:
        from timm.models import create_model
    except ImportError as exc:
        raise RuntimeError("timm with LaBraM model registration is required") from exc
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(project.training.model_init_seed))
        backbone = create_model(
            "labram_base_patch200_200",
            pretrained=False,
            num_classes=0,
            num_patches_per_channel_input=4,
            init_values=0.1,
        )
    audit = strict_load_pretrained(backbone, project.checkpoint)
    digest = torch_state_digest(backbone.state_dict())
    backbone = backbone.to(torch.device(project.device))
    for parameter in backbone.parameters():
        if parameter.is_floating_point():
            parameter.requires_grad_(True)
    return backbone, audit, digest


def create_task_classifier(
    backbone,
    num_classes: int,
    input_chans: Sequence[int],
    head_seed: int,
):
    torch, nn, _ = require_torch()

    class _TaskClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.input_chans = [int(x) for x in input_chans]
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(head_seed))
                self.head = nn.Linear(200, int(num_classes))

        def forward(self, x):
            features = self.backbone.forward_features(x, input_chans=self.input_chans)
            if features.ndim != 2 or features.shape[-1] != 200:
                raise ValueError(
                    f"Expected LaBraM pooled features [B,200], found {tuple(features.shape)}"
                )
            return self.head(features)

    return _TaskClassifier()


def extract_tsv_weights(backbone) -> Dict[str, object]:
    torch, _, _ = require_torch()
    if not hasattr(backbone, "blocks") or len(backbone.blocks) != N_BLOCKS:
        raise ValueError("Expected LaBraM base with 12 Transformer blocks")
    result = {}
    for block_index, block in enumerate(backbone.blocks):
        qkv = block.attn.qkv.weight.detach()
        if tuple(qkv.shape) != (600, 200):
            raise ValueError(f"Block {block_index}: qkv shape {tuple(qkv.shape)}")
        q, k, v = torch.split(qkv, 200, dim=0)
        values = {
            "Q": q,
            "K": k,
            "V": v,
            "O": block.attn.proj.weight.detach(),
            "fc1": block.mlp.fc1.weight.detach(),
            "fc2": block.mlp.fc2.weight.detach(),
        }
        for role, value in values.items():
            name = f"L{block_index:02d}/{role}"
            if tuple(value.shape) != EXPECTED_MODULE_SHAPES[name]:
                raise ValueError(
                    f"{name}: expected {EXPECTED_MODULE_SHAPES[name]}, found {tuple(value.shape)}"
                )
            result[name] = value.cpu().clone()
    if set(result) != set(EXPECTED_MODULE_SHAPES):
        raise AssertionError("Failed to extract the exact 72-matrix grid")
    return result


def assign_tsv_matrices(backbone, matrices: Mapping[str, np.ndarray]) -> None:
    torch, _, _ = require_torch()
    if set(matrices) != set(EXPECTED_MODULE_SHAPES):
        raise ValueError("assign_tsv_matrices requires the complete 72-matrix grid")
    with torch.no_grad():
        for block_index, block in enumerate(backbone.blocks):
            qkv = []
            for role in ("Q", "K", "V"):
                value = np.asarray(matrices[f"L{block_index:02d}/{role}"])
                qkv.append(
                    torch.as_tensor(
                        value,
                        device=block.attn.qkv.weight.device,
                        dtype=block.attn.qkv.weight.dtype,
                    )
                )
            block.attn.qkv.weight.copy_(torch.cat(qkv, dim=0))
            for role, parameter in (
                ("O", block.attn.proj.weight),
                ("fc1", block.mlp.fc1.weight),
                ("fc2", block.mlp.fc2.weight),
            ):
                parameter.copy_(
                    torch.as_tensor(
                        np.asarray(matrices[f"L{block_index:02d}/{role}"]),
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, prediction in zip(y_true.astype(int), y_pred.astype(int)):
        matrix[truth, prediction] += 1
    return matrix


def metrics_from_confusion(matrix: np.ndarray) -> Dict[str, object]:
    n = int(matrix.sum())
    accuracy = float(np.trace(matrix) / n) if n else float("nan")
    recalls = []
    f1s = []
    for index in range(matrix.shape[0]):
        tp = float(matrix[index, index])
        fn = float(matrix[index, :].sum() - matrix[index, index])
        fp = float(matrix[:, index].sum() - matrix[index, index])
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
        recalls.append(recall)
        f1s.append(f1)
    return {
        "n": n,
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "recall_per_class": recalls,
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_classifier(classifier, loader, device, n_classes: int, weights=None):
    torch, _, functional = require_torch()
    classifier.eval()
    loss_numerator = 0.0
    loss_denominator = 0.0
    truths = []
    predictions = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = classifier(x)
            per_sample = functional.cross_entropy(logits, y, weight=weights, reduction="none")
            denominator_weights = (
                torch.ones_like(y, dtype=per_sample.dtype) if weights is None else weights[y]
            )
            loss_numerator += float(per_sample.sum().item())
            loss_denominator += float(denominator_weights.sum().item())
            truths.append(y.detach().cpu().numpy())
            predictions.append(logits.argmax(dim=1).detach().cpu().numpy())
    if not truths:
        raise RuntimeError("Empty evaluation loader")
    result = metrics_from_confusion(
        confusion_matrix(np.concatenate(truths), np.concatenate(predictions), n_classes)
    )
    result["objective_loss"] = loss_numerator / max(loss_denominator, EPS)
    return result


class FunctionalEvaluator:
    """Reconstruct and evaluate candidate updates while preserving adapted context."""

    def __init__(
        self,
        project: ProjectConfig,
        run: RunRecord,
        meta: TaskMeta,
        decompositions: Mapping[Tuple[str, int, str], object],
        verify_full_replay: bool = True,
    ):
        torch, _, _ = require_torch()
        self.project = project
        self.run = run
        self.meta = meta
        self.decompositions = decompositions
        self.device = torch.device(project.device)
        self.task_data = None

        backbone, _, base_digest = build_labram_backbone(project)
        expected_digest = str(run.metadata.get("base_full_backbone_digest", ""))
        if expected_digest and base_digest != expected_digest:
            raise RuntimeError(f"{run.tag}: current W0 differs from the training W0")
        self.base_tsv = {
            key: value.numpy().astype(np.float64, copy=True)
            for key, value in extract_tsv_weights(backbone).items()
        }

        saved = load_torch_file(run.checkpoint_path)
        if not isinstance(saved, Mapping):
            raise RuntimeError(f"{run.tag}: adapted checkpoint is not a mapping")
        if str(saved.get("task")) != run.task or int(saved.get("replicate", -1)) != run.replicate:
            raise RuntimeError(f"{run.tag}: adapted checkpoint identity mismatch")
        saved_channels = tuple(int(x) for x in saved.get("input_chans", ()))
        if saved_channels != tuple(project.input_chans):
            raise RuntimeError(f"{run.tag}: checkpoint channel mapping mismatch")
        if expected_digest and str(saved.get("base_full_backbone_digest", "")) != expected_digest:
            raise RuntimeError(f"{run.tag}: checkpoint W0 digest mismatch")
        backbone.load_state_dict(saved["backbone"], strict=True)
        classifier = create_task_classifier(
            backbone, meta.n_classes, project.input_chans, head_seed=0
        ).to(self.device)
        classifier.head.load_state_dict(saved["head"], strict=True)
        self.backbone = backbone
        self.classifier = classifier
        self.adapted_tsv = {
            key: value.numpy().astype(np.float64, copy=True)
            for key, value in extract_tsv_weights(backbone).items()
        }
        self.task_data = load_task_into_memory(meta, project.training.ram_reserve_gb)
        self.loader = make_eval_loader(
            self.task_data, project.analysis.eval_batch_size
        )
        self.weights = class_weights(meta, self.device)
        stored_metrics = dict(run.metadata.get("metrics_final", {}))
        if verify_full_replay:
            self.full_metrics = evaluate_classifier(
                classifier, self.loader, self.device, meta.n_classes, self.weights
            )
            stored_bacc = stored_metrics.get("balanced_accuracy")
            if stored_bacc is not None and abs(
                float(stored_bacc) - float(self.full_metrics["balanced_accuracy"])
            ) > project.analysis.metric_tolerance:
                raise RuntimeError(f"{run.tag}: saved and replayed full bACC differ")
        else:
            if stored_metrics.get("balanced_accuracy") is None:
                raise RuntimeError(f"{run.tag}: stored final bACC is missing")
            self.full_metrics = stored_metrics

        delta = load_delta_weights(run.delta_path)
        errors = []
        for module, value in delta.items():
            denominator = max(float(np.linalg.norm(value)), EPS)
            errors.append(
                float(
                    np.linalg.norm(self.base_tsv[module] + value - self.adapted_tsv[module])
                    / denominator
                )
            )
        self.delta_checkpoint_reconstruction_error = max(errors)
        if self.delta_checkpoint_reconstruction_error > project.analysis.metric_tolerance:
            raise RuntimeError(
                f"{run.tag}: delta archive does not reconstruct adapted checkpoint matrices"
            )

    @property
    def full_bacc(self) -> float:
        return float(self.full_metrics["balanced_accuracy"])

    def evaluate_updates(self, updates: Mapping[str, np.ndarray]) -> Dict[str, object]:
        if set(updates) != set(EXPECTED_MODULE_SHAPES):
            raise ValueError("Functional evaluation requires one update for every analysed matrix")
        matrices = {
            module: self.base_tsv[module] + np.asarray(updates[module], dtype=np.float64)
            for module in EXPECTED_MODULE_SHAPES
        }
        assign_tsv_matrices(self.backbone, matrices)
        return evaluate_classifier(
            self.classifier,
            self.loader,
            self.device,
            self.meta.n_classes,
            self.weights,
        )

    def evaluate_q(self, q: float) -> Dict[str, object]:
        from .geometry import rank_for_fraction

        updates = {}
        for module, shape in EXPECTED_MODULE_SHAPES.items():
            record = self.decompositions[(self.run.task, self.run.replicate, module)]
            k = rank_for_fraction(record.rank, q)
            updates[module] = record.reconstruct_prefix(k, shape)
        return self.evaluate_updates(updates)

    def evaluate_global_order(
        self, order: Sequence[Tuple[str, int]], component_count: int
    ) -> Dict[str, object]:
        selected: Dict[str, list] = {module: [] for module in EXPECTED_MODULE_SHAPES}
        for module, index in order[: int(component_count)]:
            selected[module].append(int(index))
        updates = {}
        for module, shape in EXPECTED_MODULE_SHAPES.items():
            record = self.decompositions[(self.run.task, self.run.replicate, module)]
            updates[module] = record.reconstruct_indices(selected[module], shape)
        return self.evaluate_updates(updates)

    def close(self) -> None:
        for name in ("classifier", "backbone", "loader", "task_data"):
            if hasattr(self, name):
                delattr(self, name)
        gc.collect()
        try:
            torch, _, _ = require_torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except RuntimeError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
