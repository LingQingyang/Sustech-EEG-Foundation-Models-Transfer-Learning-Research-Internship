"""Single source of truth for the LaBraM x M3CV experiment.

The old implementation spread scientific constants across several versioned
scripts and changed some of them through module-level monkey-patching.  This
module replaces that behaviour with immutable dataclasses.  Paths may be
overridden at the command line; scientific defaults are explicit and recorded
in every output manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Tuple


TASK_ORDER: Tuple[str, ...] = ("Rest", "Motor", "P300", "SSS", "TS")
MODULE_ROLES: Tuple[str, ...] = ("Q", "K", "V", "O", "fc1", "fc2")
N_BLOCKS = 12
N_MODULES = N_BLOCKS * len(MODULE_ROLES)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    filename: str
    class_names: Mapping[int, str]

    @property
    def raw_labels(self) -> Tuple[int, ...]:
        return tuple(sorted(int(x) for x in self.class_names))

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


TASKS: Dict[str, TaskSpec] = {
    "Rest": TaskSpec(
        "Rest",
        "m3cv_resting_session1_4s.h5",
        {0: "Beg_EC (Task 1)", 2: "Beg_EO (Task 2)"},
    ),
    "Motor": TaskSpec(
        "Motor",
        "m3cv_Motor_Session1_4s.h5",
        {0: "FT", 1: "RH", 2: "LH"},
    ),
    "P300": TaskSpec(
        "P300",
        "m3cv_P300_Session1_4s.h5",
        {0: "Non-target", 1: "Target"},
    ),
    "SSS": TaskSpec(
        "SSS",
        "m3cv_SSS_Session1_4s.h5",
        {0: "SSVEP", 1: "SSAEP", 2: "SSSEP"},
    ),
    "TS": TaskSpec(
        "TS",
        "m3cv_TS_Session1_4s.h5",
        {0: "VEP", 1: "AEP", 2: "SEP"},
    ),
}


def module_names() -> Tuple[str, ...]:
    return tuple(
        f"L{block:02d}/{role}"
        for block in range(N_BLOCKS)
        for role in MODULE_ROLES
    )


def expected_module_shapes() -> Dict[str, Tuple[int, int]]:
    shape_for_role = {
        "Q": (200, 200),
        "K": (200, 200),
        "V": (200, 200),
        "O": (200, 200),
        "fc1": (800, 200),
        "fc2": (200, 800),
    }
    return {
        f"L{block:02d}/{role}": shape_for_role[role]
        for block in range(N_BLOCKS)
        for role in MODULE_ROLES
    }


EXPECTED_MODULE_SHAPES = expected_module_shapes()


def total_rank_one_components() -> int:
    """Maximum number of matrix-wise SVD components across all 72 matrices."""
    return sum(min(shape) for shape in EXPECTED_MODULE_SHAPES.values())


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    replicates: int = 2
    batch_size: int = 32
    eval_batch_size: int = 64
    backbone_lr: float = 1e-4
    head_lr: float = 1e-3
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8
    amp: bool = False
    workers: int = 0
    ram_reserve_gb: float = 4.0
    seed_base: int = 20260819
    model_init_seed: int = 314159

    def validate(self) -> None:
        if self.epochs != 20:
            raise ValueError("The reported protocol fixes the endpoint at E*=20.")
        if self.replicates != 2:
            raise ValueError("The reported protocol requires exactly two replicates.")
        if self.batch_size < 1 or self.eval_batch_size < 1:
            raise ValueError("Batch sizes must be positive.")
        if self.backbone_lr <= 0 or self.head_lr <= 0:
            raise ValueError("Learning rates must be positive.")
        if self.weight_decay != 0.0:
            raise ValueError("The reported protocol fixes weight_decay=0.")
        if self.amp:
            raise ValueError("The reported protocol fixes AMP off.")
        if self.workers != 0:
            raise ValueError("The RAM-resident deterministic loader fixes workers=0.")
        if self.ram_reserve_gb < 0:
            raise ValueError("ram_reserve_gb must be non-negative.")
        if not (0 < self.beta1 < 1 and 0 < self.beta2 < 1):
            raise ValueError("Adam betas must lie in (0,1).")
        if self.adam_eps <= 0:
            raise ValueError("adam_eps must be positive.")


@dataclass(frozen=True)
class AnalysisConfig:
    """All speed/precision choices are data, never mutable module globals."""

    profile: str = "reported"
    target_functional_retention: float = 0.99
    coarse_fractions: Tuple[float, ...] = (
        0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.0
    )
    fine_step: float = 0.02
    energy_vertical_step: float = 0.05
    bacc_vertical_step: float = 0.10
    shared_random_n: int = 128
    sti_random_n: int = 128
    random_seed: int = 20260907
    random_upper_quantile: float = 0.95
    context_pairing: str = "matched"
    ortho_rtol: float = 0.0
    metric_tolerance: float = 2e-5
    eval_batch_size: int = 128
    display_shared_quantile: float = 0.95
    save_random_draws: bool = False

    @classmethod
    def for_profile(cls, profile: str) -> "AnalysisConfig":
        if profile == "reported":
            return cls()
        if profile == "full":
            return cls(
                profile="full",
                coarse_fractions=tuple(round(i / 20.0, 8) for i in range(21)),
                fine_step=0.01,
                bacc_vertical_step=0.05,
                shared_random_n=1000,
                sti_random_n=1000,
                random_upper_quantile=0.99,
                context_pairing="all",
                eval_batch_size=64,
            )
        raise ValueError(f"Unknown analysis profile: {profile}")

    @property
    def energy_vertical_targets(self) -> Tuple[float, ...]:
        n = int(round(1.0 / self.energy_vertical_step))
        values = {round(i * self.energy_vertical_step, 8) for i in range(n + 1)}
        values.add(0.99)
        return tuple(sorted(values))

    @property
    def bacc_vertical_targets(self) -> Tuple[float, ...]:
        n = int(round(1.0 / self.bacc_vertical_step))
        return tuple(round(i * self.bacc_vertical_step, 8) for i in range(n + 1))

    def validate(self) -> None:
        if self.profile not in {"reported", "full"}:
            raise ValueError("profile must be reported or full")
        if self.context_pairing not in {"matched", "all"}:
            raise ValueError("context_pairing must be matched or all")
        if not 0 < self.target_functional_retention <= 1:
            raise ValueError("target_functional_retention must lie in (0,1]")
        if not 0 < self.fine_step < 1:
            raise ValueError("fine_step must lie in (0,1)")
        if not 0 < self.energy_vertical_step <= 1:
            raise ValueError("energy_vertical_step must lie in (0,1]")
        if not 0 < self.bacc_vertical_step <= 1:
            raise ValueError("bacc_vertical_step must lie in (0,1]")
        if (
            not self.coarse_fractions
            or self.coarse_fractions[0] != 0.0
            or self.coarse_fractions[-1] != 1.0
            or tuple(sorted(set(self.coarse_fractions))) != self.coarse_fractions
            or any(not 0 <= value <= 1 for value in self.coarse_fractions)
        ):
            raise ValueError("coarse_fractions must be unique, sorted, and span 0 to 1")
        if self.shared_random_n < 1 or self.sti_random_n < 1:
            raise ValueError("random repeat counts must be positive")
        if not 0 < self.random_upper_quantile < 1:
            raise ValueError("random_upper_quantile must lie in (0,1)")
        if not 0 < self.display_shared_quantile < 1:
            raise ValueError("display_shared_quantile must lie in (0,1)")
        if self.ortho_rtol < 0:
            raise ValueError("ortho_rtol must be non-negative")
        if self.metric_tolerance <= 0:
            raise ValueError("metric_tolerance must be positive")
        if self.eval_batch_size < 1:
            raise ValueError("eval_batch_size must be positive")


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    data_root: Path
    checkpoint: Path
    modeling_file: Path
    output_dir: Path
    device: str = "cuda"
    input_chans: Tuple[int, ...] = field(
        default_factory=lambda: tuple([0] + list(range(1, 65)))
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    def validate(self, require_paths: bool = False) -> None:
        self.training.validate()
        self.analysis.validate()
        if len(self.input_chans) != 65 or self.input_chans[0] != 0:
            raise ValueError("input_chans must be CLS=0 followed by 64 EEG positions")
        if tuple(self.input_chans[1:]) != tuple(range(1, 65)):
            raise ValueError("M3CV mapping is fixed to exact LaBraM positions 1..64")
        if require_paths:
            for path, label in (
                (self.data_root, "data root"),
                (self.checkpoint, "LaBraM checkpoint"),
                (self.modeling_file, "LaBraM modeling file"),
            ):
                if not path.exists():
                    raise FileNotFoundError(f"Missing {label}: {path}")

    def task_path(self, task: str) -> Path:
        return self.data_root / TASKS[task].filename

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        for key in ("root", "data_root", "checkpoint", "modeling_file", "output_dir"):
            payload[key] = str(payload[key])
        payload["input_chans"] = list(self.input_chans)
        return payload
