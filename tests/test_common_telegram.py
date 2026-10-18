import subprocess
import sys

import pytest

from pynight.common_telegram import matplotlib_figure_p


def test_import_does_not_load_matplotlib():
    #: In a fresh interpreter, as this one may have imported matplotlib already.
    code = (
        "import sys, pynight.common_telegram\n"
        "assert 'matplotlib' not in sys.modules, 'matplotlib was imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_matplotlib_figure_p():
    assert not matplotlib_figure_p("plot.png")
    figure_module = pytest.importorskip("matplotlib.figure")
    assert matplotlib_figure_p(figure_module.Figure())
