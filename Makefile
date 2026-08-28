.PHONY: setup lint test data train clean

setup:
	uv sync --locked

lint:
	uv run ruff check . && uv run ruff format --check .

test:
	uv run pytest -q

clean:
	rm -rf .venv .pytest_cache .ruff_cache

data:
	uv run python -m scout data

train:
	uv run python -m scout train contribution
