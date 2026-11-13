from __future__ import annotations

from pathlib import Path

from config import ProjectConfig
from runtime import SharedNamespaceRuntime


class ExperimentRunner:
    CORE_STAGES = [
        "00_config.py",
        "01_preflight.py",
        "02_targets.py",
        "03_data.py",
        "04_saved_downstream.py",
        "05_graphcast.py",
        "06_attack_core.py",
    ]
    DIAGNOSTIC_STAGE = "07_diagnostics.py"
    ATTACK_STAGE = "08_run_attack.py"
    EVAL_STAGE = "09_evaluate.py"
    AGGREGATE_STAGE = "10_aggregate.py"

    def __init__(self, config: ProjectConfig, package_root: str | Path | None = None):
        config.validate()
        self.config = config
        self.package_root = Path(package_root or Path(__file__).resolve().parent)
        self.runtime = SharedNamespaceRuntime(
            self.package_root,
            config.to_runtime_dict(),
        )
        self._core_ready = False

    def prepare(self) -> None:
        if self._core_ready:
            return
        for stage in self.CORE_STAGES:
            print(f"\n>>> {stage}")
            self.runtime.execute(stage)
        self._core_ready = True

    def diagnostic(self) -> None:
        self.prepare()
        print(f"\n>>> {self.DIAGNOSTIC_STAGE}")
        self.runtime.execute(self.DIAGNOSTIC_STAGE)

    def attack(self) -> None:
        self.prepare()
        print(f"\n>>> {self.ATTACK_STAGE}")
        self.runtime.execute(self.ATTACK_STAGE)

    def evaluate(self) -> None:
        self.prepare()
        print(f"\n>>> {self.EVAL_STAGE}")
        self.runtime.execute(self.EVAL_STAGE)

    def aggregate(self) -> None:
        self.prepare()
        print(f"\n>>> {self.AGGREGATE_STAGE}")
        self.runtime.execute(self.AGGREGATE_STAGE)

    def run(self, through: str = "aggregate", run_diagnostic: bool = False) -> None:
        order = ["prepare", "attack", "evaluate", "aggregate"]
        if through not in order:
            raise ValueError(f"through must be one of {order}")

        self.prepare()
        if run_diagnostic:
            self.diagnostic()
        if through == "prepare":
            return
        self.attack()
        if through == "attack":
            return
        self.evaluate()
        if through == "evaluate":
            return
        self.aggregate()
