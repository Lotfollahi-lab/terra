# Contributing guide

Contributions to TERRA are welcome — bug reports, bug fixes, documentation, and
new features. This guide summarises the essentials to get started.

## Development install

```bash
git clone https://github.com/Lotfollahi-lab/terra.git
cd terra
pip install -e ".[dev,test,doc]"
```

## Code style

TERRA uses [pre-commit](https://pre-commit.com/) to enforce a consistent code
style ([ruff](https://docs.astral.sh/ruff/) for linting and formatting, plus
prettier). Enable it once in your clone:

```bash
pre-commit install
```

Hooks then run automatically on every commit, fixing issues or reporting errors.

## Tests

TERRA uses [pytest](https://docs.pytest.org/). Please add tests for new
functionality. Run the suite from the repository root:

```bash
pytest
```

Continuous integration runs the tests on `dev` and on pull requests against the
minimum and a recent supported Python version, plus a pre-release-dependencies
job to catch upstream incompatibilities early.

## Documentation

Documentation is built with [Sphinx](https://www.sphinx-doc.org/) and
[MyST-NB](https://myst-nb.readthedocs.io/). Notebooks placed in `docs/notebooks`
are rendered as tutorials. Public functions and classes use
[NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).
Build the docs locally with:

```bash
cd docs
make html
open _build/html/index.html
```

## Releasing

Bump the `version` in `pyproject.toml` following
[Semantic Versioning](https://semver.org/) (MAJOR.MINOR.PATCH). Then create a
GitHub release with a `vX.Y.Z` tag — this triggers the workflow that builds and
publishes the package to PyPI.
