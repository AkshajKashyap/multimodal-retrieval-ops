import json
from pathlib import Path
import socket
import sys

import pytest

from multimodal_retrieval_ops import __version__
from multimodal_retrieval_ops.cli import main
from multimodal_retrieval_ops.portfolio_release import (
    RELEASE_VERSION,
    run_portfolio_smoke,
    write_portfolio_outputs,
)
from multimodal_retrieval_ops.release_consistency import validate_release_consistency
from test_milestone_one import run_cli

ROOT = Path(__file__).parents[1]
REQUIRED_DOCS = (
    "architecture.md",
    "model_card.md",
    "evaluation_methodology.md",
    "operations.md",
    "release_checklist.md",
    "interview_notes.md",
)


@pytest.fixture(scope="module")
def smoke_result():
    return run_portfolio_smoke(supplied_test_count=12)


def test_version_output_is_stable() -> None:
    result = run_cli("--version")
    assert result.returncode == 0
    assert result.stdout == "multimodal-retrieval-ops 1.0.0\n"
    assert __version__ == RELEASE_VERSION == "1.0.0"


def test_project_info_is_deterministic_and_complete(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["project-info"]) == 0
    information = json.loads(capsys.readouterr().out)
    assert information["name"] == "multimodal-retrieval-ops"
    assert information["version"] == "1.0.0"
    assert information["python_requirement"] == ">=3.11"
    assert information["retrieval_directions"] == ["text_to_image", "image_to_text"]
    assert information["serving_backends"] == ["FlatIP", "HNSW"]
    assert information["telemetry_schema_version"] == 1
    assert information["primary_evaluation_dataset"] == "Flickr8k official test split"
    assert information["release_limitations"]


def test_release_consistency_validator(tmp_path: Path) -> None:
    (tmp_path / "reports/portfolio").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "multimodal-retrieval-ops"\nversion = "1.0.0"\n',
        encoding="utf-8",
    )
    (tmp_path / "CITATION.cff").write_text('version: "1.0.0"\n', encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text("## [1.0.0]\n", encoding="utf-8")
    (tmp_path / "reports/portfolio/release_1.0.0.json").write_text(
        '{"release_version": "1.0.0"}\n', encoding="utf-8"
    )
    assert validate_release_consistency(tmp_path).consistent is True


def test_portfolio_smoke_is_synthetic_bidirectional_and_monitored(smoke_result) -> None:
    assert smoke_result.smoke_state == "success"
    assert smoke_result.synthetic_artifacts is True
    assert smoke_result.neural_inference_used is False
    assert smoke_result.retrieval_directions_exercised == [
        "text_to_image",
        "image_to_text",
    ]
    assert smoke_result.service_endpoints_exercised == [
        "/health",
        "/ready",
        "/retrieve/images",
        "/retrieve/captions",
    ]
    assert smoke_result.telemetry_exercised is True
    assert smoke_result.telemetry_event_count == 4
    assert smoke_result.monitoring_health_decision == "insufficient_data"


def test_portfolio_smoke_blocks_network_and_does_not_load_neural_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access is forbidden in the portfolio smoke")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    before = {name for name in sys.modules if name.startswith(("torch", "transformers"))}
    result = run_portfolio_smoke()
    after = {name for name in sys.modules if name.startswith(("torch", "transformers"))}
    assert result.smoke_state == "success"
    assert after == before


def test_portfolio_reports_are_deterministic_and_machine_independent(
    tmp_path: Path, smoke_result
) -> None:
    first_report = tmp_path / "first.md"
    first_metrics = tmp_path / "first.json"
    second_report = tmp_path / "second.md"
    second_metrics = tmp_path / "second.json"
    write_portfolio_outputs(smoke_result, first_report, first_metrics)
    write_portfolio_outputs(smoke_result, second_report, second_metrics)
    assert first_report.read_bytes() == second_report.read_bytes()
    assert first_metrics.read_bytes() == second_metrics.read_bytes()
    content = first_report.read_text(encoding="utf-8") + first_metrics.read_text(
        encoding="utf-8"
    )
    assert str(tmp_path) not in content
    assert "/home/" not in content
    assert "event_timestamp" not in content
    assert "T00:" not in content


def test_readme_links_and_required_documentation_exist() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in REQUIRED_DOCS:
        assert (ROOT / "docs" / name).is_file()
        assert f"docs/{name}" in readme


def test_ci_remains_non_neural() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert '".[dev,faiss,serve]"' in workflow
    assert ",clip" not in workflow
    assert ",hfdata" not in workflow
    assert ",train" not in workflow
    assert "torch" not in workflow.lower()
    assert "transformers" not in workflow.lower()


def test_dockerfile_packages_no_models_datasets_or_real_artifacts() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    lowered = dockerfile.lower()
    assert "python:3.11-slim" in lowered
    assert "user appuser" in lowered
    assert "torch" not in lowered
    assert "transformers" not in lowered
    assert "copy data" not in lowered
    assert "copy artifacts" not in lowered
    assert "model weight" not in lowered
    assert "healthcheck" in lowered


def test_release_files_are_present() -> None:
    for name in ("LICENSE", "CHANGELOG.md", "CITATION.cff", "CONTRIBUTING.md"):
        assert (ROOT / name).is_file()
