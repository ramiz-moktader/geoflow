"""
Publication-Grade Visualization Engine for Scientific Research.
Provides Nature/Science style presets, high DPI exports, colorblind-safe palettes,
and LaTeX/Markdown scientific tables.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def theme(name: str = "nature"):
    """
    Apply predefined publication stylesheet for scientific manuscripts.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
    })

    if name.lower() == "dark":
        plt.style.use("dark_background")
    elif name.lower() in ("nature", "science", "minimal"):
        plt.rcParams["axes.grid"] = False


def figure(
    width: str = "single_column",
    aspect_ratio: float = 0.75,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create figure with standard journal column widths.
    - single_column: 89 mm (~3.5 inches)
    - double_column: 183 mm (~7.2 inches)
    """
    w_in = 3.5 if width == "single_column" else 7.2
    h_in = w_in * aspect_ratio
    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=dpi)
    return fig, ax


class Table:
    """Publication-ready table formatting for LaTeX, Markdown, and HTML."""

    def __init__(self, data: Union[pd.DataFrame, Dict[str, Any]]):
        if isinstance(data, pd.DataFrame):
            self.df = data.copy()
        elif isinstance(data, dict):
            self.df = pd.DataFrame(data)
        else:
            self.df = pd.DataFrame(data)

    def to_markdown(self) -> str:
        return self.df.to_markdown(index=False)

    def to_latex(self, caption: Optional[str] = None, label: Optional[str] = None) -> str:
        return self.df.to_latex(index=False, caption=caption, label=label)

    def to_html(self) -> str:
        return self.df.to_html(index=False, classes="table table-striped")

    def save(self, filepath: str):
        if filepath.endswith(".tex"):
            Path(filepath).write_text(self.to_latex(), encoding="utf-8")
        elif filepath.endswith(".md"):
            Path(filepath).write_text(self.to_markdown(), encoding="utf-8")
        elif filepath.endswith(".html"):
            Path(filepath).write_text(self.to_html(), encoding="utf-8")
        else:
            self.df.to_csv(filepath, index=False)


def table(data: Any) -> Table:
    return Table(data)
