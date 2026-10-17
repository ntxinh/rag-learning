# Thin shortcuts over `uv run`. No build logic lives here.
RUN := uv run

.DEFAULT_GOAL := help

.PHONY: help
help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

.PHONY: sync
sync: ## Install/sync dependencies from uv.lock
	uv sync --all-groups

.PHONY: test
test: ## Run the test suite
	$(RUN) pytest

.PHONY: lint
lint: ## Lint with Ruff
	$(RUN) ruff check .

.PHONY: format
format: ## Format with Ruff
	$(RUN) ruff format . && $(RUN) ruff check --fix .

.PHONY: typecheck
typecheck: ## Type-check with mypy
	$(RUN) mypy src tests

.PHONY: check
check: lint typecheck test ## Lint + typecheck + tests (CI-equivalent)

.PHONY: api
api: ## Run the FastAPI dev server on :8000
	$(RUN) uvicorn rag_learning.api.app:app --reload

.PHONY: ingest
ingest: ## Ingest documents: make ingest PATH=data/documents
	$(RUN) rag ingest $(or $(PATH),data/documents)

.PHONY: ask
ask: ## Ask a question: make ask Q="what is ..."
	$(RUN) rag ask "$(Q)"

.PHONY: eval
eval: ## Run evaluation: make eval DATASET=data/evaluation/questions.jsonl
	$(RUN) rag evaluate $(or $(DATASET),data/evaluation/questions.jsonl)

.PHONY: reset-index
reset-index: ## Wipe the vector collection (guarded in CLI too)
	$(RUN) rag reset --yes

.PHONY: clean
clean: ## Delete vector DB and tool caches
	rm -rf storage/chroma .pytest_cache .mypy_cache .ruff_cache
