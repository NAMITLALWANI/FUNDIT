.PHONY: install test lint run ingest index evaluate docker-up docker-down clean

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
PYTEST ?= .venv/bin/pytest
UVICORN ?= .venv/bin/uvicorn

install:
	$(PIP) install -r requirements-dev.txt

ingest:
	$(PYTHON) scripts/ingest.py

index:
	$(PYTHON) scripts/build_index.py

evaluate:
	$(PYTHON) scripts/evaluate.py

test:
	$(PYTEST) -v

run:
	$(UVICORN) app.api.main:app --host 0.0.0.0 --port 8000 --reload

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
