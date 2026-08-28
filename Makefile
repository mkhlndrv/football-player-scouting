.PHONY: setup lint test data spike clean

setup:
	uv sync --locked

lint:
	uv run ruff check . && uv run ruff format --check .

test:
	uv run pytest -q

spike:
	for s in spike/s*.py; do echo "== $$s"; uv run python $$s || exit 1; done

clean:
	rm -rf .venv .pytest_cache .ruff_cache data/spike

data:
	uv run python -m scout data
