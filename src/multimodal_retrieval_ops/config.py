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
    baseline_index_path: Path = Path("artifacts/baseline/text_index.json")
    baseline_vocab_path: Path = Path("artifacts/baseline/vocab.json")
    baseline_report_path: Path = Path("reports/baseline_retrieval_report.md")
    baseline_metrics_path: Path = Path("reports/baseline_retrieval_metrics.json")
    multimodal_index_path: Path = Path("artifacts/baseline/multimodal_index.json")
    multimodal_report_path: Path = Path("reports/multimodal_baseline_report.md")
    multimodal_metrics_path: Path = Path("reports/multimodal_baseline_metrics.json")
    clip_index_path: Path = Path("artifacts/clip/clip_index.json")
    clip_cache_path: Path = Path("artifacts/clip/embedding_cache.json")
    clip_backend_report_path: Path = Path("reports/clip_backend_report.md")
    clip_report_path: Path = Path("reports/clip_retrieval_report.md")
    clip_metrics_path: Path = Path("reports/clip_retrieval_metrics.json")


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
        baseline_index_path=Path(
            project.get("baseline_index_path", ProjectConfig.baseline_index_path)
        ),
        baseline_vocab_path=Path(
            project.get("baseline_vocab_path", ProjectConfig.baseline_vocab_path)
        ),
        baseline_report_path=Path(
            project.get("baseline_report_path", ProjectConfig.baseline_report_path)
        ),
        baseline_metrics_path=Path(
            project.get("baseline_metrics_path", ProjectConfig.baseline_metrics_path)
        ),
        multimodal_index_path=Path(
            project.get("multimodal_index_path", ProjectConfig.multimodal_index_path)
        ),
        multimodal_report_path=Path(
            project.get("multimodal_report_path", ProjectConfig.multimodal_report_path)
        ),
        multimodal_metrics_path=Path(
            project.get("multimodal_metrics_path", ProjectConfig.multimodal_metrics_path)
        ),
        clip_index_path=Path(project.get("clip_index_path", ProjectConfig.clip_index_path)),
        clip_cache_path=Path(project.get("clip_cache_path", ProjectConfig.clip_cache_path)),
        clip_backend_report_path=Path(
            project.get("clip_backend_report_path", ProjectConfig.clip_backend_report_path)
        ),
        clip_report_path=Path(project.get("clip_report_path", ProjectConfig.clip_report_path)),
        clip_metrics_path=Path(project.get("clip_metrics_path", ProjectConfig.clip_metrics_path)),
    )
