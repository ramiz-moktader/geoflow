"""
Command Line Interface for GeoFlow.
Provides `geoflow search`, `download`, `info`, `gpu`, and `cache` commands.
"""

from __future__ import annotations
import click
import json
import shutil
from pathlib import Path

from geoflow.core.config import config
from geoflow.earthdata.search import search
from geoflow.earthdata.downloader import download


@click.group()
def cli():
    """GeoFlow: GEE-like Earth Observation & Spatial ML CLI."""
    pass


@click.command()
@click.argument("dataset")
@click.option("--bbox", help="Bounding box minx,miny,maxx,maxy")
@click.option("--start", help="Start date (YYYY-MM-DD)")
@click.option("--end", help="End date (YYYY-MM-DD)")
@click.option("--limit", default=10, help="Max results to fetch")
def search_cmd(dataset: str, bbox: str, start: str, end: str, limit: int):
    """Search NASA Earthdata granules."""
    region = [float(x) for x in bbox.split(",")] if bbox else None
    results = search(dataset, region=region, start=start, end=end, limit=limit)
    click.echo(f"Found {len(results)} granules for dataset '{dataset}':")
    for r in results:
        title = r.get("title", r.get("id"))
        click.echo(f"  - {title}")


@click.command()
def info():
    """Display GeoFlow configuration and environment information."""
    click.echo("=== GeoFlow Environment Information ===")
    click.echo(f"Default CRS  : {config.default_crs}")
    click.echo(f"Cache Dir    : {config.cache_dir}")
    click.echo(f"Backend      : {config.backend}")
    total, used, free = shutil.disk_usage(Path.cwd())
    click.echo(f"Disk Free    : {free / (1024*1024*1024):.2f} GB")


@click.command()
def gpu():
    """Check GPU acceleration availability."""
    import sys
    click.echo("=== Hardware Acceleration Status ===")
    cuda_avail = False
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        click.echo(f"PyTorch CUDA Available: {cuda_avail}")
        if cuda_avail:
            click.echo(f"Device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        click.echo("PyTorch not installed.")

    try:
        import cupy
        click.echo("CuPy Available: True")
    except ImportError:
        click.echo("CuPy not installed.")


cli.add_command(search_cmd, "search")
cli.add_command(info, "info")
cli.add_command(gpu, "gpu")


if __name__ == "__main__":
    cli()
