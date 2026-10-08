"""Configuration: the environment registry, then the registries and config files.

Keep this package free of heavy imports; the test suite imports
``lakegen.config.settings`` before anything else to know which variables to
scrub from the environment.
"""
