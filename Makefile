install:
	python -m pip install -e ".[dev]"

check:
	pytest -q
	ruff check .

test:
	pytest -q

lint:
	ruff check .
