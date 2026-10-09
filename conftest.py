import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Ensure all matplotlib figures are closed after each test to avoid memory leaks."""
    yield
    plt.close("all")
