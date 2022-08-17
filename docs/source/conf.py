# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../../rpylib"))


# def add_to_path():
#     partial_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')
#     workspace_path = os.path.abspath(partial_path)
#     assert os.path.exists(workspace_path)
#
#     projects = []
#
#     for current, dirs, c in os.walk(str(workspace_path)):
#         for dir in dirs:
#
#             project_path = os.path.join(workspace_path, dir, 'src')
#
#             if os.path.exists(project_path):
#                 projects.append(project_path)
#
#     for project_str in projects:
#         sys.path.append(project_str)
#
#
# add_to_path()

project = "rpylib"
copyright = "2022, Romain Palfray"
author = "Romain Palfray"
version = "1.0"
release = "1.0"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = []

add_module_names = False
autodoc_default_options = {
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
