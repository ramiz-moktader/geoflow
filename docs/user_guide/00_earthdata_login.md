# NASA Earthdata Login

Some GeoFlow data comes from NASA: GEDI LiDAR (`gf.GEDI`), HLS Landsat/Sentinel-2 (`gf.ImageCollection.from_earthdata`) and anything you fetch with `gf.search` and `gf.download`. NASA requires a free **Earthdata Login** account to download it. This page walks you through creating one and connecting it to GeoFlow.

!!! tip "Don't need NASA data yet?"
    Sentinel-2, Landsat and other imagery from **Microsoft Planetary Computer** (`gf.ImageCollection.from_stac(...)`) needs no account at all. The [cartography tutorial](07_cartography_tutorial.md) uses only that.

## Step 1: Create your Earthdata account

1. Go to **<https://urs.earthdata.nasa.gov/users/new>**.
2. Choose a **username** and **password**. You'll type these into GeoFlow later, so write the username down exactly.
3. Fill in your name, email, country, affiliation and study area, accept NASA's terms and click **Register for Earthdata Login**.
4. Open the confirmation email from NASA and click the activation link. **Downloads fail until the account is activated.**

## Step 2: Authorize the data centres you'll use

NASA's archives are run by separate data centres (DAACs). Each one has an "application" that must be authorized on your account before it accepts your downloads. If it isn't, downloads fail with **HTTP 401** or a **500 error from `urs.earthdata.nasa.gov/oauth/authorize`**.

| GeoFlow data | Data centre |
|---|---|
| GEDI L2A / L2B (`gf.GEDI("L2A")`, `"L2B"`) | LP DAAC |
| GEDI L4A biomass | ORNL DAAC |
| HLS Sentinel-2 / Landsat (`from_earthdata("SENTINEL-2")`) | LP DAAC |

The most reliable way to authorize one is to download a single file in your browser:

1. Sign in at **<https://urs.earthdata.nasa.gov>**.
2. In the same browser, open any data link for that data centre, for example a GEDI file URL from a GeoFlow error message or from a `gf.search(...)` result.
3. If NASA shows an **Authorize** page for the application, approve it. The file then starts downloading, and you can cancel it.

You can review and approve applications any time under **Applications → Authorized Apps** in your Earthdata profile.

## Step 3: Log in from GeoFlow

=== "Jupyter / terminal"

    ```python
    import geoflow as gf

    gf.login()  # asks for your Earthdata username and password
    ```

    GeoFlow saves the credentials to `~/.netrc` (`_netrc` on Windows) so later sessions log in without asking. Use `gf.login(persist=False)` to keep them in memory only.

=== "Google Colab"

    Colab forgets files when the runtime restarts, so the safest setup is Colab **Secrets**:

    1. Click the **🔑 key icon** in Colab's left sidebar.
    2. Add a secret named `EARTHDATA_USERNAME` with your username, and one named `EARTHDATA_PASSWORD` with your password. Turn on **Notebook access** for both.
    3. Log in at the top of your notebook:

    ```python
    import os
    from google.colab import userdata
    import geoflow as gf

    os.environ["EARTHDATA_USERNAME"] = userdata.get("EARTHDATA_USERNAME")
    os.environ["EARTHDATA_PASSWORD"] = userdata.get("EARTHDATA_PASSWORD")
    gf.login()
    ```

    Your password never appears in the notebook, so you can share it safely. Alternatively, `gf.login()` on its own prompts for both values.

=== "Scripts / servers"

    Set two environment variables before starting Python:

    ```bash
    export EARTHDATA_USERNAME="your-username"
    export EARTHDATA_PASSWORD="your-password"
    ```

    Then `gf.login()` picks them up without prompting.

!!! warning "Keep your password private"
    Never paste your password into a notebook cell, a script you share, or a GitHub repository. Use the prompt, Colab Secrets or environment variables shown above.

## Step 4: Check it works

Download one small GEDI granule over your area. GEDI files are large (often more than 1 GB), so always start with `limit=1`:

```python
aoi = gf.Rectangle([91.76, 22.27, 91.90, 22.40])  # replace with your area

gedi = (
    gf.GEDI("L2A")
    .filterBounds(aoi)
    .filterDate("2022-01-01", "2022-03-31")
    .filterQuality()
    .select(["rh98", "sensitivity"])
    .load(limit=1)
)
gdf = gedi.to_geodataframe()
print(len(gdf), "GEDI shots")
```

!!! note
    `gf.login()` returns `True` as soon as it finds credentials, without contacting NASA. A real download, like the one above, is what confirms your username, password and application authorization are right.

## Troubleshooting

| What you see | What it means | Fix |
|---|---|---|
| `HTTP 401` / "NASA Earthdata refused access" | Wrong username or password, account not activated, or the data centre's app isn't authorized | Check the activation email, do **Step 2**, then `gf.login(force=True)` and re-enter your details |
| `500 Server Error ... urs.earthdata.nasa.gov/oauth/authorize` | NASA's login server rejected the request, usually for the same reasons as a 401 | Same as above |
| "This GEDI object has no data files yet" | You filtered a `GEDI` object but never downloaded anything | Add `.load(limit=1)` at the end of the chain |
| "GEDI.load() needs a region" | No area given, which would download orbits from anywhere on Earth | Add `.filterBounds(aoi)` or pass `region=aoi` |
| "Got an HTML page instead of data" | NASA sent its login page instead of the file | Run `gf.login()` before downloading |
| Login worked before, fails now | Password changed, or a typo was saved | `gf.login(force=True)` asks again and replaces the saved entry |

To log out of the current Python session, call `gf.logout()`. To remove saved credentials, delete the `machine urs.earthdata.nasa.gov` line from `~/.netrc`.
