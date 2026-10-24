# GeoFlow Documentation

Welcome to the official documentation for **GeoFlow** (`gf`).

GeoFlow is a production-quality Python package that provides a Google Earth Engine-like developer experience (`Image`, `ImageCollection`, `FeatureCollection`, `Filter`, `Reducer`), while seamlessly integrating with the Python geospatial ecosystem (xarray, geopandas, scikit-learn). 

It specifically introduces first-class features for **Spatial Statistics**, **Leakage-Safe Spatial Machine Learning**, and **Publication Cartography**.

## Table of Contents

### User Guide
1. [Quickstart](user_guide/01_quickstart.md)
1. [NASA Earthdata Login](user_guide/00_earthdata_login.md): create an account and connect it to GeoFlow
2. [Images and Collections](user_guide/02_images_and_collections.md)
3. [Spatial Statistics](user_guide/03_spatial_statistics.md)
4. [Machine Learning and Validation](user_guide/04_machine_learning.md)
5. [Cartography and Visualization](user_guide/05_cartography.md)
6. [Comprehensive End-to-End Tutorial](user_guide/06_end_to_end_tutorial.md)
7. [Tutorial: Making Maps](user_guide/07_cartography_tutorial.md): seven publication maps from free Sentinel-2 data, step by step

### API Reference
- [Core (`gf.core`)](api_reference/core.md)
- [Raster (`gf.raster`)](api_reference/raster.md)
- [Vector & Geometry (`gf.vector` & `gf.geometry`)](api_reference/vector.md)
- [Earthdata & Datasets (`gf.earthdata`, `gf.auth`, `gf.datasets`)](api_reference/earthdata_and_datasets.md)
- [Spatial Statistics & Inference (`gf.spatial` & `gf.stats`)](api_reference/spatial.md)
- [Machine Learning & Validation (`gf.ml`, `gf.validation`, `gf.sampling`)](api_reference/ml_and_validation.md)
- [Visualization & Cartography (`gf.viz` & `gf.visualization`)](api_reference/viz.md)
- [CLI Reference (`geoflow.cli`)](api_reference/cli.md)

### Development
- [Contributor & Architecture Guide](contributing.md)
