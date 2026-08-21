PY ?= .venv/bin/python

# The interpreter used to CREATE the venv. Bare `python3` is not good enough on
# macOS, where it is still the system 3.9 — numpy needs 3.12+, pandas 3.11+ and
# streamlit 3.10+, so a 3.9 venv fails at `pip install`. Pick the newest real
# interpreter on PATH, and fall back to Homebrew's if none is linked.
BOOTSTRAP ?= $(shell for v in 3.13 3.12 3.11 3.10; do \
	command -v python$$v 2>/dev/null && break; \
	test -x /opt/homebrew/opt/python@$$v/bin/python$$v && \
		echo /opt/homebrew/opt/python@$$v/bin/python$$v && break; \
	done)

.PHONY: help setup sample index reindex stats ask ui test eval clean

help:
	@echo "make setup     create .venv and install dependencies"
	@echo "make sample    generate the sample enterprise PDFs"
	@echo "make index     build the vector index from scratch"
	@echo "make reindex   upsert new/changed documents only"
	@echo "make stats     show what is currently indexed"
	@echo "make ui        run the Streamlit app"
	@echo "make test      run the test suite"
	@echo "make eval      run the grounding benchmark"
	@echo "make clean     delete the index"

setup:
	@test -n "$(BOOTSTRAP)" || { \
		echo "No Python 3.10+ found. Install one first:"; \
		echo "    brew install python@3.12"; exit 1; }
	@echo "Creating .venv with $(BOOTSTRAP) ($$($(BOOTSTRAP) -V))"
	$(BOOTSTRAP) -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	@test -f .env || cp .env.example .env
	@echo "\nDone. Next:  make sample && make index && make ui"

sample:
	$(PY) -m scripts.generate_test_pdfs

index:
	$(PY) -m scripts.ingest --rebuild

reindex:
	$(PY) -m scripts.ingest

stats:
	$(PY) -m scripts.ingest --stats

ui:
	.venv/bin/streamlit run app.py

test:
	$(PY) -m pytest tests/ -q

eval:
	$(PY) -m scripts.evaluate

clean:
	rm -rf storage/chroma
