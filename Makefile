# Chirag -- simple install & usage
#
#   make install            set everything up and put `chirag` on PATH
#   make models             pull the Ollama models
#   make index              build the knowledge base
#   make query Q="question" ask a question
#   make chat               interactive terminal UI (TUI)
#   make serve              HTTP API server (REST + OpenAI-compatible)
#   make stats              show knowledge-base statistics
#   make test               run the test suite
#   make eval               run the evaluation harness
#   make clean              remove caches
#   make uninstall          remove the `chirag` symlink from PATH

SHELL := /bin/bash
PYTHON ?= python3
VENV   := venv
PIP    := $(VENV)/bin/pip
PY     := $(VENV)/bin/python
CHIRAG   := $(abspath $(VENV)/bin/chirag)
BIN    := $(HOME)/.local/bin
LLM_MODEL    ?= qwen3.5:9b
EMBED_MODEL  ?= nomic-embed-text

.PHONY: help install models index index-reset query chat serve stats test eval clean uninstall

help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install: ## create venv, install deps, link `chirag` into PATH
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip >/dev/null
	$(PIP) install -e '.[dev]'
	@test -f .env || { echo "[install] creating .env from .env.example"; cp .env.example .env; }
	@mkdir -p $(BIN)
	ln -sf $(CHIRAG) $(BIN)/chirag
	@echo
	@echo "[install] done."
	@echo "  command:  $(BIN)/chirag  (already on your PATH)"
	@echo "  next:     make models   ->  pull Ollama models"
	@echo "            make index    ->  build the knowledge base"
	@echo "            chirag      ->  start chatting (TUI)"

models: ## pull the Ollama embedding + chat models
	ollama pull $(EMBED_MODEL)
	ollama pull $(LLM_MODEL)

index: ## build the knowledge base (adds to existing index)
	$(CHIRAG) index-corpus

index-reset: ## rebuild the knowledge base from scratch
	$(CHIRAG) index-corpus --reset

query: ## ask a question:  make query Q="What is AES?"
	$(CHIRAG) query "$(Q)"

chat: ## launch the interactive terminal UI
	$(CHIRAG)

serve: ## run the HTTP API server (REST + OpenAI-compatible)
	$(CHIRAG) serve

stats: ## show knowledge-base statistics
	$(CHIRAG) stats

test: ## run the test suite
	$(PY) -m pytest

eval: ## run the evaluation harness
	$(CHIRAG) eval

clean: ## remove Python caches
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -not -path './venv/*' -prune -exec rm -rf {} +

uninstall: ## remove the `chirag` symlink from PATH
	rm -f $(BIN)/chirag
	@echo "[uninstall] removed $(BIN)/chirag (venv left in place)"
