# One image: the interpreter and system libraries the lockfile assumes. Data is mounted, never baked in.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock .python-version ./
COPY src ./src
RUN uv sync --locked --no-dev
COPY tests ./tests
COPY Makefile README.md ./
COPY models ./models
CMD ["uv", "run", "python", "-m", "scout", "data"]
