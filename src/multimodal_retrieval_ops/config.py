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
    schema_v2_manifest_path: Path = Path("data/processed/manifest_v2.csv")
    flickr8k_manifest_path: Path = Path("data/processed/flickr8k_manifest_v2.csv")
    benchmark_manifest_path: Path = Path("data/processed/flickr8k_benchmark_v2.csv")
    schema_v2_report_path: Path = Path("reports/schema_v2_migration_report.md")
    flickr8k_report_path: Path = Path("reports/flickr8k_dataset_report.md")
    clip_benchmark_index_path: Path = Path("artifacts/clip/flickr8k_index.json")
    clip_benchmark_cache_path: Path = Path("artifacts/clip/flickr8k_cache.json")
    clip_benchmark_report_path: Path = Path("reports/clip_real_benchmark_report.md")
    clip_benchmark_metrics_path: Path = Path("reports/clip_real_benchmark_metrics.json")
    hf_flickr8k_manifest_path: Path = Path("data/processed/hf_flickr8k_manifest_v2.csv")
    hf_flickr8k_images_path: Path = Path("data/raw/hf_flickr8k/images")
    hf_flickr8k_provenance_path: Path = Path("data/processed/hf_flickr8k_provenance.json")
    hf_flickr8k_report_path: Path = Path("reports/hf_flickr8k_dataset_report.md")
    hf_integration_cache_path: Path = Path("artifacts/clip/hf_flickr8k_integration_cache.json")
    hf_integration_index_path: Path = Path("artifacts/clip/hf_flickr8k_integration_index.json")
    hf_integration_report_path: Path = Path("reports/clip_flickr8k_integration_report.md")
    hf_integration_metrics_path: Path = Path("reports/clip_flickr8k_integration_metrics.json")
    hf_test_cache_path: Path = Path("artifacts/clip/hf_flickr8k_test_cache.json")
    hf_test_index_path: Path = Path("artifacts/clip/hf_flickr8k_test_index.json")
    hf_test_report_path: Path = Path("reports/clip_flickr8k_test_report.md")
    hf_test_metrics_path: Path = Path("reports/clip_flickr8k_test_metrics.json")
    faiss_artifacts_path: Path = Path("artifacts/faiss")
    faiss_report_path: Path = Path("reports/faiss_flat_correctness_report.md")
    faiss_metrics_path: Path = Path("reports/faiss_flat_correctness_metrics.json")
    faiss_hnsw_artifacts_path: Path = Path("artifacts/faiss_hnsw")
    faiss_hnsw_report_path: Path = Path("reports/faiss_hnsw_comparison_report.md")
    faiss_hnsw_metrics_path: Path = Path("reports/faiss_hnsw_comparison_metrics.json")
    retrieval_service_report_path: Path = Path("reports/retrieval_service_report.md")
    retrieval_service_metrics_path: Path = Path("reports/retrieval_service_metrics.json")
    text_inference_service_report_path: Path = Path("reports/text_inference_service_report.md")
    text_inference_service_metrics_path: Path = Path("reports/text_inference_service_metrics.json")


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
        schema_v2_manifest_path=Path(
            project.get("schema_v2_manifest_path", ProjectConfig.schema_v2_manifest_path)
        ),
        flickr8k_manifest_path=Path(
            project.get("flickr8k_manifest_path", ProjectConfig.flickr8k_manifest_path)
        ),
        benchmark_manifest_path=Path(
            project.get("benchmark_manifest_path", ProjectConfig.benchmark_manifest_path)
        ),
        schema_v2_report_path=Path(
            project.get("schema_v2_report_path", ProjectConfig.schema_v2_report_path)
        ),
        flickr8k_report_path=Path(
            project.get("flickr8k_report_path", ProjectConfig.flickr8k_report_path)
        ),
        clip_benchmark_index_path=Path(
            project.get("clip_benchmark_index_path", ProjectConfig.clip_benchmark_index_path)
        ),
        clip_benchmark_cache_path=Path(
            project.get("clip_benchmark_cache_path", ProjectConfig.clip_benchmark_cache_path)
        ),
        clip_benchmark_report_path=Path(
            project.get("clip_benchmark_report_path", ProjectConfig.clip_benchmark_report_path)
        ),
        clip_benchmark_metrics_path=Path(
            project.get("clip_benchmark_metrics_path", ProjectConfig.clip_benchmark_metrics_path)
        ),
        hf_flickr8k_manifest_path=Path(
            project.get("hf_flickr8k_manifest_path", ProjectConfig.hf_flickr8k_manifest_path)
        ),
        hf_flickr8k_images_path=Path(
            project.get("hf_flickr8k_images_path", ProjectConfig.hf_flickr8k_images_path)
        ),
        hf_flickr8k_provenance_path=Path(
            project.get("hf_flickr8k_provenance_path", ProjectConfig.hf_flickr8k_provenance_path)
        ),
        hf_flickr8k_report_path=Path(
            project.get("hf_flickr8k_report_path", ProjectConfig.hf_flickr8k_report_path)
        ),
        hf_integration_cache_path=Path(
            project.get("hf_integration_cache_path", ProjectConfig.hf_integration_cache_path)
        ),
        hf_integration_index_path=Path(
            project.get("hf_integration_index_path", ProjectConfig.hf_integration_index_path)
        ),
        hf_integration_report_path=Path(
            project.get("hf_integration_report_path", ProjectConfig.hf_integration_report_path)
        ),
        hf_integration_metrics_path=Path(
            project.get("hf_integration_metrics_path", ProjectConfig.hf_integration_metrics_path)
        ),
        hf_test_cache_path=Path(
            project.get("hf_test_cache_path", ProjectConfig.hf_test_cache_path)
        ),
        hf_test_index_path=Path(
            project.get("hf_test_index_path", ProjectConfig.hf_test_index_path)
        ),
        hf_test_report_path=Path(
            project.get("hf_test_report_path", ProjectConfig.hf_test_report_path)
        ),
        hf_test_metrics_path=Path(
            project.get("hf_test_metrics_path", ProjectConfig.hf_test_metrics_path)
        ),
        faiss_artifacts_path=Path(
            project.get("faiss_artifacts_path", ProjectConfig.faiss_artifacts_path)
        ),
        faiss_report_path=Path(
            project.get("faiss_report_path", ProjectConfig.faiss_report_path)
        ),
        faiss_metrics_path=Path(
            project.get("faiss_metrics_path", ProjectConfig.faiss_metrics_path)
        ),
        faiss_hnsw_artifacts_path=Path(
            project.get("faiss_hnsw_artifacts_path", ProjectConfig.faiss_hnsw_artifacts_path)
        ),
        faiss_hnsw_report_path=Path(
            project.get("faiss_hnsw_report_path", ProjectConfig.faiss_hnsw_report_path)
        ),
        faiss_hnsw_metrics_path=Path(
            project.get("faiss_hnsw_metrics_path", ProjectConfig.faiss_hnsw_metrics_path)
        ),
        retrieval_service_report_path=Path(
            project.get(
                "retrieval_service_report_path", ProjectConfig.retrieval_service_report_path
            )
        ),
        retrieval_service_metrics_path=Path(
            project.get(
                "retrieval_service_metrics_path", ProjectConfig.retrieval_service_metrics_path
            )
        ),
        text_inference_service_report_path=Path(
            project.get(
                "text_inference_service_report_path",
                ProjectConfig.text_inference_service_report_path,
            )
        ),
        text_inference_service_metrics_path=Path(
            project.get(
                "text_inference_service_metrics_path",
                ProjectConfig.text_inference_service_metrics_path,
            )
        ),
    )
