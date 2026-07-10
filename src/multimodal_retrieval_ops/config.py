"""Typed project configuration loaded from TOML."""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ProjectConfig:
    """Paths used by the Milestone 1 commands."""

    manifest_path: Path = Path("data/processed/demo_manifest.csv")
    report_path: Path = Path("reports/demo_manifest_summary.md")


def load_config(path: Path | None = None) -> ProjectConfig:
    """Load configuration, falling back to deterministic project defaults."""
    if path is None:
        return ProjectConfig()
    with path.open("rb") as config_file:
        raw = tomllib.load(config_file)
    project = raw.get("project", {})
    return ProjectConfig(
        manifest_path=Path(project.get("manifest_path", ProjectConfig.manifest_path)),
        report_path=Path(project.get("report_path", ProjectConfig.report_path)),
    )
