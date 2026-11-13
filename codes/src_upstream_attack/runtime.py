from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any


class SharedNamespaceRuntime:
    """Execute stage scripts in one shared namespace.

    The original experiment was a notebook with many functions resolving module-level
    state (models, tensors, paths and experiment settings). Executing stage modules in a
    shared namespace preserves those exact semantics while giving the project a class-driven
    driver, configuration layer, resumable stages and CLI selection.
    """

    def __init__(self, package_root: str | Path, project_config: dict[str, Any]):
        self.package_root = Path(package_root)
        self.stage_root = self.package_root / "stages"
        self.namespace: dict[str, Any] = {
            "__name__": "__nips26_runtime__",
            "__package__": None,
            "PROJECT_CONFIG": dict(project_config),
        }
        self.executed: list[str] = []

    @property
    def view(self):
        return MappingProxyType(self.namespace)

    def execute(self, stage_file: str) -> None:
        path = self.stage_root / stage_file
        if not path.exists():
            raise FileNotFoundError(path)
        source = path.read_text()
        self.namespace["__file__"] = str(path)
        code = compile(source, str(path), "exec")
        exec(code, self.namespace, self.namespace)
        self.executed.append(stage_file)

    def get(self, name: str, default=None):
        return self.namespace.get(name, default)
