# Contributing

Thanks for your interest in `clinical-llm`. This is currently a personal
research project, but contributions, issues, and discussion are welcome.

## Development setup

```bash
git clone https://github.com/z-awan-lab/clinical-llm.git
cd clinical-llm
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install   # optional
```

## Before opening a PR

- `ruff check src tests` — passes
- `black --check src tests` — passes
- `pytest tests/ -v` — all green
- Add tests for new behaviour
- Update `docs/` if you changed an interface or added a feature

## Issues

If you spot a bug or have a feature suggestion, please open an issue with:

- What you tried
- What you expected
- What actually happened
- A minimal example, if possible

## Data and ethics

Never include real clinical data — even small samples — in issues, PRs, or
code. Use the synthetic generator for reproduction cases.
