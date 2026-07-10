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

clip-benchmark:
	@test -n "$(FLICKR8K_IMAGES_DIR)" || (echo "Set FLICKR8K_IMAGES_DIR" && exit 2)
	@test -n "$(FLICKR8K_CAPTIONS_FILE)" || (echo "Set FLICKR8K_CAPTIONS_FILE" && exit 2)
	multimodal-retrieval-ops ingest-flickr8k --images-dir "$(FLICKR8K_IMAGES_DIR)" --captions-file "$(FLICKR8K_CAPTIONS_FILE)"
	multimodal-retrieval-ops create-benchmark-subset --max-images 1000 --seed 42
	multimodal-retrieval-ops evaluate-clip-benchmark --model-name openai/clip-vit-base-patch32 --device cpu --batch-size 16
	multimodal-retrieval-ops search-clip --query "red car" --model-name openai/clip-vit-base-patch32 --device cpu
	multimodal-retrieval-ops evaluate-clip --model-name openai/clip-vit-base-patch32 --device cpu --batch-size 4
	multimodal-retrieval-ops build-clip-index --model-name openai/clip-vit-base-patch32 --device cpu --batch-size 4
