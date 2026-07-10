"""Typed project configuration loaded from TOML."""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ProjectConfig:
    """Paths used by the project commands."""

    manifest_path: Path = Path("data/processed/demo_manifest.csv")
    report_path: Path = Path("reports/demo_manifest_summary.md")
    fixture_path: Path = Path("tests/fixtures/local_dataset")
    ingested_manifest_path: Path = Path("data/processed/local_fixture_manifest.csv")
    dataset_manifest_path: Path = Path("data/processed/dataset_manifest.csv")
    inspection_report_path: Path = Path("reports/dataset_inspection_report.md")


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
        fixture_path=Path(project.get("fixture_path", ProjectConfig.fixture_path)),
        ingested_manifest_path=Path(
            project.get("ingested_manifest_path", ProjectConfig.ingested_manifest_path)
        ),
        dataset_manifest_path=Path(
            project.get("dataset_manifest_path", ProjectConfig.dataset_manifest_path)
        ),
        inspection_report_path=Path(
            project.get("inspection_report_path", ProjectConfig.inspection_report_path)
        ),
    )
