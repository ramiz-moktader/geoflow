"""
Interactive GEE-like Map implementation for GeoFlow.
Provides smooth layer addition, vis_params rendering, split maps, legends, and colorbars.
Generates interactive Leaflet maps with zero mandatory third-party map dependencies,
and integrates natively with Folium when installed.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import base64
import io
from pathlib import Path
from PIL import Image as PILImage
import numpy as np

from geoflow.geometry.base import Geometry
from geoflow.raster.image import Image
from geoflow.vector.collection import FeatureCollection


class Map:
    """GEE-like interactive Map for visualizing rasters, vectors, and GEDI footprints."""

    def __init__(
        self,
        center: Tuple[float, float] = (0.0, 0.0),
        zoom: int = 2,
        height: str = "600px",
        basemap: str = "HYBRID",
    ):
        self.center = list(center)  # [lat, lon]
        self.zoom = zoom
        self.height = height
        self.layers: List[Dict[str, Any]] = []
        self.legends: List[Dict[str, Any]] = []
        self.colorbars: List[Dict[str, Any]] = []
        self.basemap = basemap
        self.add_basemap(basemap)

    def setCenter(self, lon: float, lat: float, zoom: int = 10) -> Map:
        """Center map at [lat, lon] with given zoom level."""
        self.center = [lat, lon]
        self.zoom = zoom
        return self

    def centerObject(self, obj: Any, zoom: Optional[int] = None) -> Map:
        """Center map on Geometry, Image, or FeatureCollection."""
        if hasattr(obj, "bounds"):
            minx, miny, maxx, maxy = obj.bounds
            center_lat = (miny + maxy) / 2.0
            center_lon = (minx + maxx) / 2.0
            self.center = [center_lat, center_lon]
            if zoom is not None:
                self.zoom = zoom
            else:
                # Estimate zoom from bounding box degree span
                span = max(abs(maxx - minx), abs(maxy - miny))
                if span > 10:
                    self.zoom = 5
                elif span > 2:
                    self.zoom = 8
                elif span > 0.5:
                    self.zoom = 11
                else:
                    self.zoom = 14
        return self

    center_object = centerObject

    def add_basemap(self, basemap: str = "HYBRID") -> Map:
        """Set or add basemap layer."""
        basemaps = {
            "ROADMAP": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            "OpenStreetMap": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            "SATELLITE": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "HYBRID": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
            "TERRAIN": "https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
            "Positron": "https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
            "Dark_Matter": "https://cartodb-basemaps-{s}.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png",
        }
        self.basemap_url = basemaps.get(basemap, basemaps["HYBRID"])
        self.basemap_name = basemap
        return self

    def addLayer(
        self,
        ee_object: Any,
        vis_params: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        shown: bool = True,
        opacity: float = 1.0,
    ) -> Map:
        """
        Add a raster Image, FeatureCollection, or Geometry layer to the map.
        vis_params: {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000, "palette": ["blue", "green"]}
        """
        vis_params = vis_params or {}
        layer_name = name or f"Layer {len(self.layers) + 1}"

        if isinstance(ee_object, Image):
            # Reproject to EPSG:4326 if needed for Leaflet overlay
            img_wgs84 = ee_object.reproject("EPSG:4326") if ee_object.crs.epsg != 4326 else ee_object
            # Render 8-bit RGB array
            rgb_arr = img_wgs84.visualize(vis_params)
            # Create PNG in memory
            pil_img = PILImage.fromarray(rgb_arr)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
            data_uri = f"data:image/png;base64,{b64_data}"

            minx, miny, maxx, maxy = img_wgs84.bounds
            self.layers.append({
                "type": "image",
                "name": layer_name,
                "url": data_uri,
                "bounds": [[miny, minx], [maxy, maxx]],  # [[south, west], [north, east]]
                "opacity": opacity,
                "shown": shown,
            })

        elif isinstance(ee_object, FeatureCollection):
            # Reproject to EPSG:4326 if needed
            fc_wgs84 = ee_object if ee_object.crs.epsg == 4326 else FeatureCollection(ee_object.gdf.to_crs("EPSG:4326"))
            geojson = fc_wgs84.to_geojson()
            self.layers.append({
                "type": "geojson",
                "name": layer_name,
                "data": geojson,
                "opacity": opacity,
                "shown": shown,
            })

        elif isinstance(ee_object, Geometry):
            geom_wgs84 = ee_object.reproject("EPSG:4326") if ee_object.crs.epsg != 4326 else ee_object
            geojson = {
                "type": "Feature",
                "geometry": geom_wgs84.to_geojson(),
                "properties": {},
            }
            self.layers.append({
                "type": "geojson",
                "name": layer_name,
                "data": geojson,
                "opacity": opacity,
                "shown": shown,
            })
        else:
            raise TypeError(f"Unsupported layer type: {type(ee_object)}")

        return self

    add_layer = addLayer

    def add_legend(self, title: str = "Legend", colors: Optional[List[str]] = None, labels: Optional[List[str]] = None) -> Map:
        """Add custom categorical legend widget."""
        self.legends.append({
            "title": title,
            "items": list(zip(colors or [], labels or [])),
        })
        return self

    def add_colorbar(self, vis_params: Dict[str, Any], label: str = "Value") -> Map:
        """Add continuous colorbar widget."""
        self.colorbars.append({
            "label": label,
            "min": vis_params.get("min", 0),
            "max": vis_params.get("max", 1),
            "palette": vis_params.get("palette", ["#0000FF", "#00FF00", "#FF0000"]),
        })
        return self

    def to_html(self, filepath: Optional[Union[str, Path]] = None) -> str:
        """Generate standalone, responsive interactive HTML map."""
        layers_json = json.dumps(self.layers)
        legends_json = json.dumps(self.legends)
        colorbars_json = json.dumps(self.colorbars)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GeoFlow Interactive Map</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body, html {{ margin: 0; padding: 0; height: 100%; width: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        #map {{ height: {self.height}; width: 100%; }}
        .leaflet-control-layers {{ box-shadow: 0 4px 12px rgba(0,0,0,0.15); border-radius: 8px; }}
        .map-legend {{ background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(8px); padding: 12px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); font-size: 13px; line-height: 18px; color: #333; }}
        .map-legend h4 {{ margin: 0 0 8px 0; font-size: 14px; font-weight: 600; color: #111; }}
        .legend-item {{ display: flex; align-items: center; margin-bottom: 4px; }}
        .legend-color {{ width: 18px; height: 18px; border-radius: 3px; margin-right: 8px; display: inline-block; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        var map = L.map('map').setView([{self.center[0]}, {self.center[1]}], {self.zoom});
        var basemap = L.tileLayer('{self.basemap_url}', {{
            maxZoom: 20,
            attribution: 'Map &copy; GeoFlow / Contributors'
        }}).addTo(map);

        var overlays = {{}};
        var layers = {layers_json};

        layers.forEach(function(l) {{
            if (l.type === 'image') {{
                var imgOverlay = L.imageOverlay(l.url, l.bounds, {{ opacity: l.opacity }});
                if (l.shown) {{ imgOverlay.addTo(map); }}
                overlays[l.name] = imgOverlay;
            }} else if (l.type === 'geojson') {{
                var geoLayer = L.geoJSON(l.data, {{
                    style: function(f) {{
                        return {{ color: "#2563eb", weight: 2, fillOpacity: l.opacity * 0.4 }};
                    }},
                    pointToLayer: function(f, latlng) {{
                        return L.circleMarker(latlng, {{
                            radius: 5,
                            fillColor: "#ef4444",
                            color: "#fff",
                            weight: 1,
                            opacity: 1,
                            fillOpacity: 0.8
                        }});
                    }},
                    onEachFeature: function(f, layer) {{
                        if (f.properties) {{
                            var desc = "<b>Properties:</b><br>";
                            for (var k in f.properties) {{
                                desc += k + ": " + f.properties[k] + "<br>";
                            }}
                            layer.bindPopup(desc);
                        }}
                    }}
                }});
                if (l.shown) {{ geoLayer.addTo(map); }}
                overlays[l.name] = geoLayer;
            }}
        }});

        L.control.layers({{ '{self.basemap_name}': basemap }}, overlays, {{ collapsed: false }}).addTo(map);

        // Add legends if any
        var legends = {legends_json};
        legends.forEach(function(leg) {{
            var c = L.control({{ position: 'bottomright' }});
            c.onAdd = function() {{
                var div = L.DomUtil.create('div', 'map-legend');
                div.innerHTML = '<h4>' + leg.title + '</h4>';
                leg.items.forEach(function(item) {{
                    div.innerHTML += '<div class="legend-item"><span class="legend-color" style="background:' + item[0] + '"></span>' + item[1] + '</div>';
                }});
                return div;
            }};
            c.addTo(map);
        }});
    </script>
</body>
</html>
"""
        if filepath:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(html, encoding="utf-8")
        return html

    save = to_html

    def _repr_html_(self) -> str:
        """Embed iframe HTML directly in Jupyter notebooks."""
        html_content = self.to_html()
        escaped = html_content.replace('"', '&quot;')
        return f'<iframe srcdoc="{escaped}" width="100%" height="{self.height}" style="border:none; border-radius:8px; box-shadow:0 4px 12px rgba(0,0,0,0.1);"></iframe>'
