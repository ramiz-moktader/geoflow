# Vector API Reference

The `geoflow.vector` and `geoflow.geometry` modules handle spatial geometries and vector data.

## Geometries

- `gf.Point([lon, lat])`: A single coordinate pair.
- `gf.Polygon(coords)`: A closed geometry.
- `gf.Rectangle([minx, miny, maxx, maxy])`: A bounding box.
- `gf.Geometry.from_file(path)`: Loads any supported vector format (GeoJSON, shapefile) into a geometry.

## `gf.FeatureCollection`

Represents a collection of spatial features (polygons, points) with properties, backed by `GeoPandas`.

### Initialization
```python
gf.FeatureCollection(data) # From GeoJSON, shapefile, or GeoDataFrame
```

### Core Methods
- `.filterBounds(geometry)`
- `.filterMetadata(property, operator, value)`
- `.select(properties)`
- `.buffer(distance)`
- `.centroid()`
- `.simplify(tolerance)`
- `.spatial_join(other)`

### Raster Intersections
- `.sample_raster(image, scale)`: Samples raster values at feature locations (Zonal stats / Point sampling).

### Cartography
- `.plot_map(**kwargs)`: Plots the vector data using the cartography engine (identical signature to `Image.plot_map()`).
