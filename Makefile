install:
	python -m pip install -e ".[dev]"

check:
	pytest -q
	ruff check .

test:
	pytest -q

lint:
	ruff check .

clip-smoke:
	multimodal-retrieval-ops clip-backend-info --model-name openai/clip-vit-base-patch32 --device cpu
	multimodal-retrieval-ops build-clip-index --model-name openai/clip-vit-base-patch32 --device cpu --batch-size 4
	multimodal-retrieval-ops search-clip --query "red car" --model-name openai/clip-vit-base-patch32 --device cpu
	multimodal-retrieval-ops evaluate-clip --model-name openai/clip-vit-base-patch32 --device cpu --batch-size 4
	multimodal-retrieval-ops build-clip-index --model-name openai/clip-vit-base-patch32 --device cpu --batch-size 4
