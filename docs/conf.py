# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))
# Make the in-repo packages importable for autodoc (src layout) even if the
# package is only installed --no-deps on the docs builder.
sys.path.insert(0, str(HERE.parent / "src"))


# -- Project information -----------------------------------------------------

# NOTE: If you installed your project in editable mode, this might be stale.
#       If this is the case, reinstall it to refresh the metadata
info = metadata("terra-st")
project_name = info.get("Name", "terra-st")
project = "TERRA"
# Docs footer / copyright authors. The full project author list lives in
# pyproject.toml (and is unaffected by this).
author = "Sebastian Birk and Mohammad Vali Sanian"
copyright = f"{datetime.now():%Y}, {author}."
version = info.get("Version", "0.0.0")
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
repository_url = urls["Source"]

# The full version, including alpha/beta/rc tags
release = info["Version"]

bibtex_bibfiles = ["references.bib"]
templates_path = ["_templates"]
nitpicky = True  # Warn about broken links
needs_sphinx = "4.0"

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "Lotfollahi-lab",  # Username
    "github_repo": "terra",  # Repo name
    "github_version": "main",  # Version
    "conf_py_path": "/docs/",  # Path in the checkout to the docs root
}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxext.opengraph",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]

# Heavy / GPU-only / unavailable-on-RTD runtime dependencies. autodoc imports
# the package to introspect signatures and docstrings; these are mocked so the
# docs build needs neither a GPU nor the full scientific stack. NOTE: torch is
# intentionally NOT mocked -- it is installed (CPU build) via
# docs/requirements.txt, because public functions are decorated with
# @torch.inference_mode()/@torch.no_grad() and classes subclass nn.Module, so a
# mocked torch would erase their signatures.
autodoc_mock_imports = [
    "anndata",
    "scanpy",
    "squidpy",
    "scipy",
    "sklearn",
    "skmisc",
    "matplotlib",
    "tqdm",
    "requests",
    "datasets",
    "transformers",
    "peft",
    "scib_metrics",
    "pyensembl",
    "cellphonedb",
    "omnipath",
    "wandb",
    "leidenalg",
    "rapids_singlecell",
    "cuml",
    "cudf",
    "cugraph",
    "cuvs",
    "flash_attn",
    "einops",
    "timm",
]

autosummary_generate = True
autodoc_member_order = "groupwise"
default_role = "literal"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = False  # returns are rendered by the typed_returns extension
napoleon_use_param = True
myst_heading_anchors = 6  # create anchors for h1-h6
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto")
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
typehints_defaults = "braces"
# The return type is shown in the Returns bullet (typed_returns extension);
# don't also emit a separate "Return type:" from the annotation.
typehints_document_rtype = False

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
# Make the "Download source file" button download the file (e.g. the .ipynb
# notebook) rather than opening it inline.
html_js_files = ["js/download-source.js"]
html_logo = "_static/terra_logo.png"
html_favicon = "_static/favicon.png"

html_title = "TERRA"

# Canonical URL + OpenGraph. On Read the Docs, READTHEDOCS_CANONICAL_URL holds
# the served version's base URL; it is empty locally (no canonical/OG URLs).
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
ogp_site_url = html_baseurl
ogp_image = "_static/terra_logo.png"

html_theme_options = {
    "repository_url": repository_url,
    "repository_branch": "main",
    "use_repository_button": True,
    "use_download_button": True,
    # Show a "launch on Colab" button on the tutorial notebook pages.
    "launch_buttons": {
        "colab_url": "https://colab.research.google.com",
        "notebook_interface": "jupyterlab",
    },
    "path_to_docs": "docs/",
    "navigation_with_keys": False,
}

pygments_style = "default"

nitpick_ignore = [
    # External / mocked types with no resolvable cross-reference target
    # (their packages are mocked for the docs build or use abbreviated paths).
    ("py:class", "datasets.Dataset"),
    ("py:class", "datasets.arrow_dataset.Dataset"),
    ("py:class", "np.ndarray"),
    ("py:class", "anndata.AnnData"),
    ("py:class", "ad.AnnData"),
    ("py:class", "pandas.core.frame.DataFrame"),
    ("py:class", "pandas.DataFrame"),
    ("py:class", "terra.datasets.cell_datasets.CellBaseDataset"),
    ("py:data", "typing.Union"),
]
