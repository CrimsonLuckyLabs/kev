PYTHON ?= python3.12
PIP ?= $(PYTHON) -m pip

.PHONY: install test demo lint serve

install:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest -m "not gpu"

demo:
	$(PYTHON) -m kev.cli demo --model mock

lint:
	$(PYTHON) -m ruff check src tests examples scripts
	$(PYTHON) -m mypy src/kev

serve:
	$(PYTHON) -m kev.cli serve --model mock --port 8787
