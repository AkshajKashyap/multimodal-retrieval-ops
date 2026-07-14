# Release Checklist

- [x] Version consistency implemented and locally verified
- [x] Unit and integration tests
- [x] Ruff lint
- [x] `git diff --check`
- [ ] CI workflow execution (Python 3.11 workflow is configured but has not run remotely)
- [x] Synthetic portfolio smoke
- [x] Deterministic tracked reports
- [x] No tracked files above 10 MB
- [x] No obvious secrets in release surfaces
- [x] No absolute local paths in tracked reports or documentation
- [x] Optional dependencies remain isolated and model-free under normal checks
- [x] README documentation links
- [x] MIT license
- [x] 1.0.0 changelog
- [x] Citation metadata
- [ ] Docker smoke (not run because Docker is unavailable in the current WSL environment)
- [ ] Git tag
- [ ] GitHub release

Unchecked items require an external tool or human release action and are not claimed complete.
