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

clip-benchmark:
	@test -n "$(FLICKR8K_IMAGES_DIR)" || (echo "Set FLICKR8K_IMAGES_DIR" && exit 2)
	@test -n "$(FLICKR8K_CAPTIONS_FILE)" || (echo "Set FLICKR8K_CAPTIONS_FILE" && exit 2)
	multimodal-retrieval-ops ingest-flickr8k --images-dir "$(FLICKR8K_IMAGES_DIR)" --captions-file "$(FLICKR8K_CAPTIONS_FILE)"
	multimodal-retrieval-ops create-benchmark-subset --max-images 1000 --seed 42
	multimodal-retrieval-ops evaluate-clip-benchmark --model-name openai/clip-vit-base-patch32 --device cpu --batch-size 16

hf-flickr8k-ingest:
	multimodal-retrieval-ops ingest-hf-flickr8k --dataset-name jxie/flickr8k

clip-flickr8k-smoke:
	multimodal-retrieval-ops evaluate-clip-flickr8k --split test --max-images 100 --model-name openai/clip-vit-base-patch32 --device cpu --batch-size 16

clip-flickr8k-benchmark:
	multimodal-retrieval-ops evaluate-clip-flickr8k --split test --model-name openai/clip-vit-base-patch32 --device cpu --batch-size 16

faiss-check:
	pytest -q tests/test_milestone_seven_a.py

faiss-flat-eval:
	multimodal-retrieval-ops evaluate-faiss-flat

faiss-hnsw-check:
	pytest -q tests/test_milestone_seven_b.py

faiss-hnsw-eval:
	multimodal-retrieval-ops evaluate-faiss-hnsw

retrieval-service-check:
	pytest -q tests/test_milestone_seven_c.py

retrieval-service-smoke:
	multimodal-retrieval-ops retrieval-service-smoke --backend flat

serve-retrieval-flat:
	multimodal-retrieval-ops serve-retrieval --backend flat

serve-retrieval-hnsw:
	multimodal-retrieval-ops serve-retrieval --backend hnsw --ef-search 64

text-inference-check:
	pytest -q tests/test_milestone_eight_a.py

text-inference-smoke-flat:
	multimodal-retrieval-ops retrieval-service-smoke --backend flat --enable-text-inference --local-files-only

text-inference-smoke-hnsw:
	multimodal-retrieval-ops retrieval-service-smoke --backend hnsw --ef-search 64 --enable-text-inference --local-files-only

image-inference-check:
	pytest -q tests/test_milestone_eight_b.py

image-inference-smoke-flat:
	multimodal-retrieval-ops retrieval-service-smoke --backend flat --enable-image-inference --local-files-only

image-inference-smoke-hnsw:
	multimodal-retrieval-ops retrieval-service-smoke --backend hnsw --ef-search 64 --enable-image-inference --local-files-only
