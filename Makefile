# CryptoRAG -- simple install & usage
#
#   make install            set everything up and put `cryptorag` on PATH
#   make models             pull the Ollama models
#   make index              build the knowledge base
#   make query Q="question" ask a question
#   make chat               interactive terminal UI (TUI)
#   make serve              HTTP API server (REST + OpenAI-compatible)
#   make stats              show knowledge-base statistics
#   make test               run the test suite
#   make eval               run the evaluation harness
#   make clean              remove caches
#   make uninstall          remove the `cryptorag` symlink from PATH

SHELL := /bin/bash
PYTHON ?= python3
VENV   := venv
PIP    := $(VENV)/bin/pip
PY     := $(VENV)/bin/python
CRAG   := $(abspath $(VENV)/bin/cryptorag)
BIN    := $(HOME)/.local/bin
LLM_MODEL    ?= qwen3.5:9b
EMBED_MODEL  ?= nomic-embed-text

.PHONY: help install models index index-reset query chat serve stats test eval clean uninstall

help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install: ## create venv, install deps, link `cryptorag` into PATH
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip >/dev/null
	$(PIP) install -e '.[dev]'
	@test -f .env || { echo "[install] creating .env from .env.example"; cp .env.example .env; }
	@mkdir -p $(BIN)
	ln -sf $(CRAG) $(BIN)/cryptorag
	@echo
	@echo "[install] done."
	@echo "  command:  $(BIN)/cryptorag  (already on your PATH)"
	@echo "  next:     make models   ->  pull Ollama models"
	@echo "            make index    ->  build the knowledge base"
	@echo "            cryptorag      ->  start chatting (TUI)"

models: ## pull the Ollama embedding + chat models
	ollama pull $(EMBED_MODEL)
	ollama pull $(LLM_MODEL)

index: ## build the knowledge base (adds to existing index)
	$(CRAG) index-corpus

index-reset: ## rebuild the knowledge base from scratch
	$(CRAG) index-corpus --reset

query: ## ask a question:  make query Q="What is AES?"
	$(CRAG) query "$(Q)"

chat: ## launch the interactive terminal UI
	$(CRAG)

serve: ## run the HTTP API server (REST + OpenAI-compatible)
	$(CRAG) serve

stats: ## show knowledge-base statistics
	$(CRAG) stats

test: ## run the test suite
	$(PY) -m pytest

eval: ## run the evaluation harness
	$(CRAG) eval

clean: ## remove Python caches
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -not -path './venv/*' -prune -exec rm -rf {} +

uninstall: ## remove the `cryptorag` symlink from PATH
	rm -f $(BIN)/cryptorag
	@echo "[uninstall] removed $(BIN)/cryptorag (venv left in place)"
