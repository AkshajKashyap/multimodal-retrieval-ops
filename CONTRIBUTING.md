# Contributing

Use Python 3.11 or newer and create an isolated environment:

```bash
python -m pip install -e ".[dev,faiss,serve]"
make check
```

Optional extras are `clip` for neural encoding, `hfdata` for opt-in dataset ingestion, and `train`
for the bounded adapter implementation. Unit tests must not download datasets or model weights;
inject small deterministic fixtures instead. Run `pytest -q`, `ruff check .`, and
`git diff --check` before opening a pull request.

Reports checked into the repository must be deterministic and omit timestamps, machine paths,
secrets, and unsupported claims. Do not commit raw datasets, embedding caches, FAISS binaries,
model checkpoints, telemetry logs, or other large generated artifacts. Pull requests should state
their scope, verification commands, dependency impact, and whether any tracked scientific decision
or metric changes (normally it should not).
