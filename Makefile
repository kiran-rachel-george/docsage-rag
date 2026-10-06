.PHONY: install index run ui eval test lint docker
install:
	pip install -e ".[dev]"
index:
	python -m docsage.ingest.build_index
run:
	uvicorn docsage.api.main:app --reload
ui:
	streamlit run app/streamlit_app.py
eval:
	python eval/run_eval.py --all
test:
	pytest
lint:
	ruff check .
docker:
	docker build -t docsage-rag .
