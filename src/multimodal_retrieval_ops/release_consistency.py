"""Cross-file release-version validation for deterministic release checks."""

from dataclasses import dataclass
import json
from pathlib import Path
import re
import tomllib

from . import __version__
from .portfolio_release import RELEASE_VERSION


@dataclass(frozen=True)
class ReleaseConsistencyResult:
    expected_version: str
    pyproject_version: str
    cli_version: str
    citation_version: str
    changelog_version: str
    portfolio_report_version: str
    consistent: bool


def _required_match(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"{label} release version is missing")
    return match.group(1)


def validate_release_consistency(root: Path = Path(".")) -> ReleaseConsistencyResult:
    """Verify every public release surface agrees on the authoritative version."""
    with (root / "pyproject.toml").open("rb") as project_file:
        pyproject_version = str(tomllib.load(project_file)["project"]["version"])
    citation_version = _required_match(
        r'^version:\s*["\']?([^"\'\s]+)',
        (root / "CITATION.cff").read_text(encoding="utf-8"),
        "CITATION.cff",
    )
    changelog_version = _required_match(
        r"^## \[([^]]+)]",
        (root / "CHANGELOG.md").read_text(encoding="utf-8"),
        "CHANGELOG.md",
    )
    portfolio = json.loads(
        (root / "reports/portfolio/release_1.0.0.json").read_text(encoding="utf-8")
    )
    result = ReleaseConsistencyResult(
        expected_version=RELEASE_VERSION,
        pyproject_version=pyproject_version,
        cli_version=__version__,
        citation_version=citation_version,
        changelog_version=changelog_version,
        portfolio_report_version=str(portfolio.get("release_version", "")),
        consistent=False,
    )
    values = (
        result.pyproject_version,
        result.cli_version,
        result.citation_version,
        result.changelog_version,
        result.portfolio_report_version,
    )
    return ReleaseConsistencyResult(
        **{**result.__dict__, "consistent": all(value == RELEASE_VERSION for value in values)}
    )
