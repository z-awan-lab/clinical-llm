FROM python:3.11-slim

WORKDIR /app

# System deps for scientific Python (lightweight here; expand if needed).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install the package (no GPU deps in the default image).
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[dev]"

# Copy the rest of the project.
COPY tests ./tests
COPY configs ./configs

# Default command runs the test suite.
CMD ["pytest", "tests/", "-v"]
