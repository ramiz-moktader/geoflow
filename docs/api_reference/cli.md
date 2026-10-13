# CLI Reference (`geoflow.cli`)

GeoFlow provides a command-line interface (`geoflow`) for querying NASA Earthdata granules, inspecting environment configurations, and checking hardware acceleration status without writing Python scripts.

## Installation

When GeoFlow is installed via `pip`, the `geoflow` CLI binary is automatically registered in your environment PATH:

```bash
geoflow --help
```

---

## Commands

### `geoflow search`

Search NASA Earthdata Common Metadata Repository (CMR) directly from your terminal.

```bash
geoflow search DATASET [OPTIONS]
```

#### Arguments
- `DATASET`: Target dataset name or alias (e.g. `SENTINEL-2`, `LANDSAT-8`, `GEDI/L2A`, `MODIS`).

#### Options
- `--bbox TEXT`: Geographic bounding box as `minx,miny,maxx,maxy` (e.g. `91.7,22.2,92.0,22.5`).
- `--start TEXT`: Start date formatted as `YYYY-MM-DD`.
- `--end TEXT`: End date formatted as `YYYY-MM-DD`.
- `--limit INTEGER`: Maximum number of granule records to return (default: `10`).

#### Example
```bash
geoflow search SENTINEL-2 --bbox "91.7,22.2,92.0,22.5" --start "2023-01-01" --end "2023-01-15" --limit 5
```

---

### `geoflow info`

Displays GeoFlow system environment, active CRS, cache paths, and available local disk storage.

```bash
geoflow info
```

#### Example Output
```text
=== GeoFlow Environment Information ===
Default CRS  : EPSG:4326
Cache Dir    : C:\Users\user\.geoflow\cache
Backend      : auto
Disk Free    : 142.30 GB
```

---

### `geoflow gpu`

Inspects PyTorch CUDA GPU availability, device name, and CuPy hardware acceleration.

```bash
geoflow gpu
```

#### Example Output
```text
=== Hardware Acceleration Status ===
PyTorch CUDA Available: True
Device: NVIDIA GeForce RTX 4090
CuPy Available: True
```
