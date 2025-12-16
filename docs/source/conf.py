import os
import sys

# If you installed the package in editable mode (`uv sync`), Sphinx can often find it.
# However, pointing explicitly to 'src' is the most reliable method.
sys.path.insert(0, os.path.abspath('../../src'))

project = 'Tenso'
copyright = '2025, Khushiyant'
author = 'Khushiyant'
release = '0.5.1'

# Mock heavy dependencies so docs can build without installing CUDA/Torch
autodoc_mock_imports = [
    "cupy", 
    "torch", 
    "numpy"  # Optional: speeds up build if you don't need numpy internals
]

# Extensions
extensions = [
    'sphinx.ext.autodoc',      # Core library for html generation
    'sphinx.ext.napoleon',     # Parses the Google-style args in your code
    'sphinx.ext.viewcode',     # Adds "View Source" links
    'sphinx_autodoc_typehints', # Uses your python type hints automatically
    'myst_parser'  
]

templates_path = ['_templates']
exclude_patterns = []

# Theme configuration
html_static_path = ['_static']
html_theme = 'furo'

# Optional: Furo specific customization
html_theme_options = {
    "source_repository": "https://github.com/Khushiyant/tenso",
    "source_branch": "main",
    "source_directory": "docs/source/",
}
myst_heading_anchors = 3