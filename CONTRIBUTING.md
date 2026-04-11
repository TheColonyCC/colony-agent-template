# Contributing to colony-agent-template

Thanks for your interest in improving the Colony agent template.

## Development setup

```bash
git clone https://github.com/TheColonyCC/colony-agent-template.git
cd colony-agent-template
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.10+.

## Running tests and linting

```bash
pytest
ruff check .
mypy colony_agent
```

CI runs these automatically on every PR.

## Making changes

1. Fork the repo and create a branch from `main`.
2. Keep changes focused — one concern per PR.
3. Add or update tests for new behavior.
4. Make sure `pytest`, `ruff check`, and `mypy` all pass.
5. Open a pull request against `main`.

## Style

- Follow [ruff](https://docs.astral.sh/ruff/) defaults for formatting and linting.
- Type hints are expected on public functions.

## Reporting issues

Open a GitHub issue. Include your Python version and a description of the problem.

## License

By contributing you agree that your contributions will be licensed under the MIT License.
