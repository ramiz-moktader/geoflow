import geoflow as gf

aoi = gf.Rectangle([91.7, 22.2, 91.8, 22.3]) # Very small area to test

try:
    print("Fetching STAC collection...")
    # Load Sentinel-2 L2A from Microsoft Planetary Computer
    collection = gf.ImageCollection.from_stac(
        collection="sentinel-2-l2a",
        catalog="planetary-computer",
        geometry=aoi,
        start="2023-01-01",
        end="2023-01-10",
        limit=5,
        bands=["B04", "B03", "B02"] # RGB
    )
    
    print(f"Loaded {len(collection)} images lazyily from the cloud!")
    
    for img in collection:
        print(img)
        print(f"Data shape (bands, y, x): {img._data.shape}")
        
except Exception as e:
    print(f"STAC Error: {e}")
