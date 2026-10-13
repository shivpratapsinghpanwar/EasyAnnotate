"""Entry point for the frozen desktop build.

Everything lives in :mod:`data_factory.desktop`, so a pip install gets the
same window from ``easyannotate app``; this file exists only because
PyInstaller wants a script, not a module, to start from.
"""

from data_factory.desktop import main

if __name__ == "__main__":
    raise SystemExit(main())
