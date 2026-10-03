"""Unified command-line entry point for the clean LaBraM x M3CV workflow."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .config import AnalysisConfig, ProjectConfig, TrainingConfig
from .geometry import synthetic_self_test
from .io import (
    artifact_hashes,
    audit_delta_archives,
    discover_runs,
    dump_json,
    ensure_dir,
    load_decompositions,
    load_json,
    log,
    sha256_file,
)


COMMANDS = (
    "selftest",
    "preflight",
    "train",
    "audit",
    "spectra",
    "heatmap",
    "sti",
    "plot",
    "all",
)


def _path(value: object, base: Path) -> Path:
    text = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(text)
    return path if path.is_absolute() else (base / path).resolve()


def _known_dataclass_values(cls, values: Mapping[str, Any]) -> Dict[str, Any]:
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} config keys: {unknown}")
    return dict(values)


def _load_config_mapping(path: Optional[Path]) -> tuple[Dict[str, Any], Path]:
    if path is None:
        return {}, Path.cwd()
    path = Path(path).expanduser().resolve()
    payload = load_json(path)
    if not isinstance(payload, Mapping):
        raise TypeError("Configuration root must be a JSON object")
    return dict(payload), path.parent


def build_project_config(args: argparse.Namespace) -> ProjectConfig:
    payload, base = _load_config_mapping(args.config)
    unknown_top = sorted(
        set(payload)
        - {
            "root",
            "data_root",
            "checkpoint",
            "modeling_file",
            "output_dir",
            "device",
            "input_chans",
            "training",
            "analysis",
        }
    )
    if unknown_top:
        raise ValueError(f"Unknown project config keys: {unknown_top}")

    analysis_payload = dict(payload.get("analysis", {}))
    profile = args.profile or str(analysis_payload.pop("profile", "reported"))
    analysis = AnalysisConfig.for_profile(profile)
    analysis_payload = _known_dataclass_values(AnalysisConfig, analysis_payload)
    if "coarse_fractions" in analysis_payload:
        analysis_payload["coarse_fractions"] = tuple(
            float(x) for x in analysis_payload["coarse_fractions"]
        )
    analysis = replace(analysis, **analysis_payload)
    analysis_overrides = {
        "shared_random_n": args.shared_random_n,
        "sti_random_n": args.sti_random_n,
        "eval_batch_size": args.analysis_eval_batch_size,
    }
    analysis = replace(
        analysis,
        **{key: value for key, value in analysis_overrides.items() if value is not None},
    )

    training_payload = _known_dataclass_values(
        TrainingConfig, dict(payload.get("training", {}))
    )
    training = TrainingConfig(**training_payload)

    def configured_path(argument, key: str, default: object) -> Path:
        if argument is not None:
            return _path(argument, Path.cwd())
        return _path(payload.get(key, default), base)

    root = configured_path(args.root, "root", "artifacts")
    data_root = configured_path(args.data_root, "data_root", "data")
    checkpoint = configured_path(
        args.checkpoint, "checkpoint", "checkpoints/labram-base.pth"
    )
    modeling_file = configured_path(
        args.modeling_file, "modeling_file", "modeling_finetune.py"
    )
    if args.output_dir is not None:
        output_dir = _path(args.output_dir, Path.cwd())
    elif payload.get("output_dir") is not None:
        output_dir = _path(payload["output_dir"], base)
    else:
        output_dir = root / "analysis"
    device = args.device or str(payload.get("device", "cuda"))
    input_chans = tuple(
        int(x) for x in payload.get("input_chans", tuple([0] + list(range(1, 65))))
    )
    project = ProjectConfig(
        root=root,
        data_root=data_root,
        checkpoint=checkpoint,
        modeling_file=modeling_file,
        output_dir=output_dir,
        device=device,
        input_chans=input_chans,
        training=training,
        analysis=analysis,
    )
    project.validate(require_paths=False)
    return project


def _write_audit(project: ProjectConfig, runs) -> Dict[str, object]:
    delta = audit_delta_archives(runs)
    protocol_fingerprints = {
        str(run.metadata.get("protocol_fingerprint", "")) for run in runs
    }
    w0_digests = {
        str(run.metadata.get("base_full_backbone_digest", "")) for run in runs
    }
    if len(protocol_fingerprints) != 1 or "" in protocol_fingerprints:
        raise RuntimeError("The ten runs do not share one protocol fingerprint")
    if len(w0_digests) != 1 or "" in w0_digests:
        raise RuntimeError("The ten runs do not share one W0 digest")
    payload = {
        "status": "passed",
        "n_runs": len(runs),
        "run_grid": [[run.task, run.replicate] for run in runs],
        "protocol_fingerprint": next(iter(protocol_fingerprints)),
        "base_full_backbone_digest": next(iter(w0_digests)),
        "delta_archive_audit": delta,
        "run_artifact_sha256": {
            run.tag: {
                "metadata": sha256_file(run.metadata_path),
                "checkpoint": sha256_file(run.checkpoint_path),
                "delta": sha256_file(run.delta_path),
                "success": sha256_file(run.success_path),
            }
            for run in runs
        },
    }
    ensure_dir(project.output_dir)
    path = project.output_dir / "run_audit.json"
    dump_json(path, payload)
    dump_json(
        project.output_dir / "_AUDIT_SUCCESS.json",
        {"status": "passed", "audit_sha256": sha256_file(path)},
    )
    return payload


def execute(args: argparse.Namespace) -> Dict[str, object]:
    if args.command == "selftest":
        result = synthetic_self_test()
        return {"status": "passed", "geometry": result}

    project = build_project_config(args)
    if args.print_config:
        print(json.dumps(project.to_dict(), ensure_ascii=False, indent=2))

    if args.command == "preflight":
        from .training import run_preflight

        return run_preflight(project, force=args.force)
    if args.command == "train":
        from .training import train_all

        return train_all(
            project,
            resume=args.resume,
            overwrite=args.overwrite,
            force_preflight=args.force,
        )

    if args.command == "plot":
        from .plotting import plot_all

        return plot_all(project)

    if args.command == "all":
        from .heatmap import generate_heatmap
        from .plotting import plot_all
        from .spectra import generate_spectra
        from .sti import generate_sti
        from .training import train_all

        train_all(
            project,
            resume=args.resume,
            overwrite=args.overwrite,
            force_preflight=args.force,
        )
        runs = discover_runs(project.root)
        audit = _write_audit(project, runs)
        decompositions = load_decompositions(runs)
        spectra = generate_spectra(
            project,
            runs,
            group="all",
            resume=args.resume,
            decompositions=decompositions,
        )
        heatmap = generate_heatmap(project, runs)
        sti = generate_sti(
            project, runs, resume=args.resume, decompositions=decompositions
        )
        figures = plot_all(project)
        manifest = {
            "version": "clean-1.0",
            "status": "passed",
            "audit": audit,
            "spectra": spectra,
            "heatmap": heatmap,
            "sti": sti,
            "figures": figures,
            "artifact_sha256": artifact_hashes(project.output_dir),
        }
        path = project.output_dir / "complete_manifest.json"
        dump_json(path, manifest)
        dump_json(
            project.output_dir / "_ALL_SUCCESS.json",
            {"status": "passed", "manifest_sha256": sha256_file(path)},
        )
        return manifest

    runs = discover_runs(project.root, validate_hashes=not args.skip_hash_validation)
    if args.command == "audit":
        return _write_audit(project, runs)
    if args.command == "heatmap":
        from .heatmap import generate_heatmap

        return generate_heatmap(project, runs)

    decompositions = load_decompositions(runs)
    if args.command == "spectra":
        from .spectra import generate_spectra

        return generate_spectra(
            project,
            runs,
            group=args.group,
            resume=args.resume,
            decompositions=decompositions,
        )
    if args.command == "sti":
        from .sti import generate_sti

        return generate_sti(
            project, runs, resume=args.resume, decompositions=decompositions
        )
    raise AssertionError(args.command)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="labram-tsv",
        description="Clean LaBraM x M3CV fine-tuning and six-spectrum analysis",
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", type=Path, help="JSON configuration file")
    parser.add_argument("--root", type=Path, help="training artifact root")
    parser.add_argument("--data-root", type=Path, help="directory containing five M3CV H5 files")
    parser.add_argument("--checkpoint", type=Path, help="pretrained LaBraM checkpoint")
    parser.add_argument("--modeling-file", type=Path, help="LaBraM model registration Python file")
    parser.add_argument("--output-dir", type=Path, help="analysis output directory")
    parser.add_argument("--device", help="PyTorch device, e.g. cuda or cpu")
    parser.add_argument("--profile", choices=("reported", "full"))
    parser.add_argument("--group", choices=("main", "projection", "all"), default="all")
    parser.add_argument("--shared-random-n", type=int)
    parser.add_argument("--sti-random-n", type=int)
    parser.add_argument("--analysis-eval-batch-size", type=int)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume only scientifically signature-compatible stages",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an incomplete run directory")
    parser.add_argument("--force", action="store_true", help="recompute preflight audit")
    parser.add_argument(
        "--skip-hash-validation",
        action="store_true",
        help="diagnostic only: skip declared run hashes during discovery",
    )
    parser.add_argument("--print-config", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        result = execute(args)
    except KeyboardInterrupt:
        log("Interrupted")
        return 130
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
