"""Sphinx configuration for the public 4Denoise guide."""

from importlib.metadata import PackageNotFoundError, version as package_version
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "4Denoise"
author = "4Denoise contributors"
try:
    release = package_version("fourdenoise-main")
except PackageNotFoundError:
    release = "development"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
root_doc = "index"
exclude_patterns = ["README.md", "_build"]
myst_heading_anchors = 3
myst_enable_extensions = ["colon_fence"]
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
napoleon_google_docstring = True

# Keep network validation bounded without hiding broken destinations.
linkcheck_timeout = 15
linkcheck_retries = 2
linkcheck_workers = 2
linkcheck_rate_limit_timeout = 30


def _github_file_check_uri(app, uri):
    """Check repository file existence without GitHub's HTML-page rate limits.

    Only the checker's URI changes; readers still see the normal GitHub link.
    Missing files still return HTTP 404. Fragment/query links are left intact
    because raw content cannot validate HTML anchors or page-specific options.
    """
    prefix = "https://github.com/mirelesadan/4Denoise/blob/"
    if uri.lower().startswith(prefix.lower()) and "#" not in uri and "?" not in uri:
        return "https://raw.githubusercontent.com/mirelesadan/4Denoise/" + uri[len(prefix):]
    return None


def _require_verified_external_links(app, exception):
    """Do not let HTTP 503 responses silently pass as ignored links."""
    if exception is not None or app.builder.name != "linkcheck":
        return
    from sphinx.util import logging

    logger = logging.getLogger(__name__)
    report = Path(app.outdir) / "output.json"
    for line in report.read_text(encoding="utf-8").splitlines():
        result = json.loads(line)
        if result["status"] == "ignored" and result["uri"].startswith(("http://", "https://")):
            logger.warning("External link was not verified: %s (%s)",
                           result["uri"], result.get("info", "ignored"))
            app.statuscode = 1


def setup(app):
    app.connect("linkcheck-process-uri", _github_file_check_uri)
    app.connect("build-finished", _require_verified_external_links)

html_theme = "furo"
html_title = "4Denoise documentation"
html_baseurl = "https://mirelesadan.github.io/4Denoise/"
html_show_sourcelink = False
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#075f62",
        "color-brand-content": "#075f62",
    },
    "dark_css_variables": {
        "color-brand-primary": "#80d4ca",
        "color-brand-content": "#80d4ca",
    },
}
