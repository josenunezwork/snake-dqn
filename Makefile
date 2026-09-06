# snake-dqn — common tasks. Override the interpreter with `make PY=python test`.
PY ?= ./venv/bin/python
SRC := src web/backend

.PHONY: help install install-dev test test-fast test-web lint format web build-web clean

help:
	@echo "make install      install runtime deps"
	@echo "make install-dev  install dev deps + pre-commit hooks"
	@echo "make test         run the full Python test suite"
	@echo "make test-fast    run tests in parallel, skip slow"
	@echo "make test-web     run the frontend unit tests (vitest)"
	@echo "make lint         black --check + isort --check + flake8"
	@echo "make format       auto-format (isort + black)"
	@echo "make build-web    build the React frontend"
	@echo "make web          build + serve the web app on :8000"
	@echo "make clean        remove project caches and generated builds"

install:
	$(PY) -m pip install -r requirements.txt

install-dev:
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pre_commit install

test:
	$(PY) -m pytest -q

test-fast:
	$(PY) -m pytest -q -n auto -m "not slow"

test-web:
	cd web/frontend && npm ci && npm test

lint:
	$(PY) -m black --check --line-length 100 $(SRC)
	$(PY) -m isort --profile black --check-only $(SRC)
	$(PY) -m flake8 $(SRC)

format:
	$(PY) -m isort --profile black $(SRC)
	$(PY) -m black --line-length 100 $(SRC)

build-web:
	cd web/frontend && npm install && npm run build

web: build-web
	$(PY) web/serve.py

clean:
	find src tests web/backend -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf __pycache__ web/__pycache__ .pytest_cache .mypy_cache build snake_dqn.egg-info \
		web/frontend/dist web/frontend/.vite web/frontend/node_modules/.vite node_modules/.vite
