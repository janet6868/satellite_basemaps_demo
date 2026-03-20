"""
══════════════════════════════════════════════════════════════════
  SATELLITE BASEMAPS FOR SCIENCE STORYTELLING
  LIM Weekly Meeting · March  20 2026
══════════════════════════════════════════════════════════════════

  pip install streamlit folium streamlit-folium plotly
              geopandas pandas numpy shapely requests
              contextily matplotlib earthengine-api

  optional:
      pip install kaleido

  run:
      streamlit run streamlit_app.py

  notes:
  - Project CSVs are expected in ./data as YYYY-MM-DD.csv
  - Real Sentinel-2 RGB is pulled from Google Earth Engine
  - No cloud filtering and no cloud masking are applied to Sentinel-2
  - MODIS/CORINE/GEDI are assumed to already be present in your project CSVs
"""

from __future__ import annotations

import io
import hashlib
from pathlib import Path

import folium
from folium.plugins import Draw, Fullscreen, HeatMap, MarkerCluster, MeasureControl, MiniMap
from shapely.geometry import Point, shape as shapely_shape
from streamlit_folium import st_folium

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

try:
    import ee
    HAS_EE = True
except ImportError:
    ee = None
    HAS_EE = False

try:
    import contextily as ctx
    import matplotlib.pyplot as plt
    HAS_CTX = True
except ImportError:
    ctx = None
    plt = None
    HAS_CTX = False

try:
    import kaleido  # noqa: F401
    HAS_KALEIDO = True
except ImportError:
    HAS_KALEIDO = False


# ══════════════════════════════════════════════════════════════════
# CONFIG + STYLING
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Satellite Basemaps · LIM Weekly Meeting",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
html, body, [class*="css"] {
    font-family: Inter, system-ui, sans-serif;
    background: #f3efe7;
    color: #1f2937;
}
.stApp, [data-testid="stAppViewContainer"] {
    background: #f3efe7;
}
section[data-testid="stSidebar"] {
    background: #e8e1d6 !important;
    border-right: 1px solid #d8cec0;
}
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #ddd2c3;
    border-radius: 10px;
    padding: 10px 14px;
}
.ch-pill {
    display:inline-block;
    padding:4px 12px;
    border-radius:999px;
    background:#14532d;
    color:white;
    font-size:0.72rem;
    letter-spacing:0.08em;
    text-transform:uppercase;
    margin-bottom:8px;
}
.k-card {
    background:#ffffff;
    border:1px solid #ddd2c3;
    border-left:4px solid #14532d;
    border-radius:10px;
    padding:14px 18px;
    margin:10px 0 16px;
    line-height:1.7;
    font-size:0.92rem;
}
.story-badge {
    display:inline-block;
    background:#ece5d8;
    border:1px solid #d2c7b5;
    border-radius:8px;
    padding:6px 10px;
    font-size:0.78rem;
    margin:6px 0 12px;
}
.draw-info {
    background:#eefaf1;
    border:1px solid #8fd5a1;
    border-radius:10px;
    padding:10px 14px;
    margin:10px 0 14px;
    font-size:0.88rem;
}
</style>
""",
    unsafe_allow_html=True,
)

DATA_DIR = Path(__file__).resolve().parent / "data"

GERMANY_GEOJSON = {
    "type": "Feature",
    "properties": {"name": "Germany"},
    "geometry": {
        "type": "Polygon",
        "coordinates": [[
            [6.117, 51.900], [6.390, 52.140], [7.000, 52.380], [7.220, 53.200],
            [7.893, 53.744], [8.670, 53.943], [9.838, 54.832], [10.072, 55.056],
            [10.937, 55.395], [12.519, 54.470], [13.926, 53.935], [14.414, 53.283],
            [14.685, 52.089], [14.607, 51.745], [15.017, 51.107], [14.570, 50.922],
            [13.338, 50.733], [12.966, 50.484], [12.240, 50.267], [12.416, 49.969],
            [13.816, 48.766], [13.504, 48.588], [12.852, 48.124], [13.028, 47.637],
            [12.932, 47.468], [12.062, 47.672], [11.106, 47.396], [10.454, 47.556],
            [10.178, 47.272], [9.596, 47.525], [8.523, 47.830], [8.317, 47.614],
            [7.593, 47.578], [7.466, 47.621], [7.118, 47.750], [6.900, 48.178],
            [6.658, 49.012], [6.186, 49.463], [6.242, 49.902], [6.043, 50.128],
            [5.867, 50.469], [6.015, 51.165], [6.223, 51.360], [6.117, 51.900],
        ]],
    },
}
GERMANY_CENTER = (51.165, 10.451)
GERMANY_ZOOM = 6

TILE_LAYERS = {
    "OpenStreetMap": {
        "tiles": "OpenStreetMap",
        "attr": "© OpenStreetMap",
    },
    "CartoDB Positron": {
        "tiles": "CartoDB positron",
        "attr": "© CartoDB",
    },
    "CartoDB Dark": {
        "tiles": "CartoDB dark_matter",
        "attr": "© CartoDB",
    },
    "Esri Satellite": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "Esri, Maxar, Earthstar Geographics",
    },
    "Esri Topographic": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        "attr": "Esri, HERE, Garmin, Intermap, USGS",
    },
    "Esri NatGeo": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}",
        "attr": "Esri, National Geographic",
    },
}

# RS_OVERLAYS = {
#     "Sentinel-2 Cloudless 2021 (EOX)": {
#         "tiles": "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2021_3857/default/g/{z}/{y}/{x}.jpg",
#         "attr": "Sentinel-2 cloudless · EOX IT Services",
#     },
#     "MODIS Terra True Color (NASA GIBS)": {
#         "tiles": "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/2023-08-15/GoogleMapsCompatible/{z}/{y}/{x}.jpg",
#         "attr": "NASA GIBS / MODIS Terra",
#     },
#     "MODIS NDVI 8-day (NASA GIBS)": {
#         "tiles": "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_NDVI_8Day/default/2023-08-13/GoogleMapsCompatible/{z}/{y}/{x}.png",
#         "attr": "NASA GIBS / MODIS Terra NDVI",
#     },
#     "CORINE Land Cover 2018 (EEA)": {
#         "tiles": "https://image.discomap.eea.europa.eu/arcgis/rest/services/Corine/CLC2018_WM/MapServer/tile/{z}/{y}/{x}",
#         "attr": "EEA CORINE 2018",
#     },
#      "ESA WorldCover 2020": {
#         "kind": "wms",
#         "url": "https://services.terrascope.be/wms/v2",
#         "layers": "WORLDCOVER_2020_MAP",
#         "attr": "© ESA WorldCover 2020 / Contains modified Copernicus Sentinel data",
#         "fmt": "image/png",
#         "transparent": True,
#     },
#     "ESA WorldCover 2021": {
#         "kind": "wms",
#         "url": "https://services.terrascope.be/wms/v2",
#         "layers": "WORLDCOVER_2021_MAP",
#         "attr": "© ESA WorldCover 2021 / Contains modified Copernicus Sentinel data",
#         "fmt": "image/png",
#         "transparent": True,
#     },
#     "ESA WorldCover 2020 RGB Composite": {
#         "kind": "wms",
#         "url": "https://services.terrascope.be/wms/v2",
#         "layers": "WORLDCOVER_2020_S2_TCC",
#         "attr": "© ESA WorldCover 2020 / Sentinel-2 composite",
#         "fmt": "image/png",
#         "transparent": False,
#     },
# }
RS_OVERLAYS = {
    "ESA WorldCover 2021 Classification": {
        "kind": "wms",
        "url": "https://services.terrascope.be/wms/v2",
        "layers": "WORLDCOVER_2021_MAP",
        "attr": "ESA WorldCover 2021",
        "fmt": "image/png",
        "transparent": True,
    },
    "ESA WorldCover 2021 RGB": {
        "kind": "wms",
        "url": "https://services.terrascope.be/wms/v2",
        "layers": "WORLDCOVER_2021_S2_TCC",
        "attr": "ESA WorldCover 2021 Sentinel-2 RGB composite",
        "fmt": "image/png",
        "transparent": False,
    },
}
PLOTLY_BASEMAPS = {
    "CartoDB Positron": "carto-positron",
    "CartoDB Dark": "carto-darkmatter",
    "OpenStreetMap": "open-street-map",
    "White Background": "white-bg",
}

BASE_COLOR_MAP = {
    "Coniferous forest": "#14532d",
    "Mixed forest": "#2f855a",
    "Broadleaved forest": "#52b788",
    "Natural grassland": "#84cc16",
    "Pastures": "#bef264",
    "Arable land": "#d97706",
    "Permanent crops": "#f59e0b",
    "Heterogeneous agri.": "#e9b44c",
    "Urban (continuous)": "#7f1d1d",
    "Urban (discontinuous)": "#ef4444",
    "Industrial / commercial": "#dc2626",
    "Construction sites": "#b45309",
    "Water bodies": "#2563eb",
    "Transitional woodland": "#6ee7b7",
    "Moors & heathland": "#7c3aed",
    "Unknown": "#9ca3af",
    "Forest": "#166534",
    "Cropland": "#f59e0b",
    "Urban": "#ef4444",
    "Wetland": "#3b82f6",
    "Grassland": "#84cc16",
}
ESA_WORLDCOVER_LEGEND = {
    "Tree cover": "#006400",
    "Shrubland": "#ffbb22",
    "Grassland": "#ffff4c",
    "Cropland": "#f096ff",
    "Built-up": "#fa0000",
    "Bare / sparse vegetation": "#b4b4b4",
    "Snow and ice": "#f0f0f0",
    "Permanent water bodies": "#0064c8",
    "Herbaceous wetland": "#0096a0",
    "Mangroves": "#00cf75",
    "Moss and lichen": "#fae6a0",
}
REAL_NUMERIC_CANDIDATES = [
    "cloud_fraction",
    "Cloud_Optical_Thickness",
    "canopy_height_m",
    "ndvi",
    "cloud_top_height_m",
    "Sur_Reflectance_NIR",
    "Sur_Reflectance_Red",
]


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def code_block(code: str) -> None:
    st.code(code.strip(), language="python")


def make_folium_map(base_key: str, center=GERMANY_CENTER, zoom: int = GERMANY_ZOOM) -> folium.Map:
    cfg = TILE_LAYERS[base_key]
    m = folium.Map(location=list(center), zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles=cfg["tiles"],
        attr=cfg["attr"],
        name=base_key,
        overlay=False,
        control=True,
    ).add_to(m)
    Fullscreen(position="topright").add_to(m)
    MeasureControl(position="topright").add_to(m)
    return m


def add_germany_boundary(m: folium.Map, name: str = "Germany boundary") -> None:
    folium.GeoJson(
        GERMANY_GEOJSON,
        name=name,
        style_function=lambda _: {
            "color": "#e63946",
            "weight": 2.2,
            "fillColor": "#e63946",
            "fillOpacity": 0.03,
            "dashArray": "6 4",
        },
    ).add_to(m)
# def add_overlay_to_map(m: folium.Map, overlay_name: str, opacity: float = 1.0) -> None:
#     ov = RS_OVERLAYS[overlay_name]

#     if ov.get("kind") == "wms":
#         folium.WmsTileLayer(
#             url=ov["url"],
#             layers=ov["layers"],
#             name=overlay_name,
#             attr=ov["attr"],
#             fmt=ov.get("fmt", "image/png"),
#             transparent=ov.get("transparent", True),
#             overlay=True,
#             control=True,
#             opacity=opacity,
#         ).add_to(m)
#     else:
#         folium.TileLayer(
#             tiles=ov["tiles"],
#             attr=ov["attr"],
#             name=overlay_name,
#             overlay=True,
#             control=True,
#             opacity=opacity,
#         ).add_to(m)
def add_overlay_to_map(m: folium.Map, overlay_name: str, opacity: float = 1.0) -> None:
    ov = RS_OVERLAYS[overlay_name]

    if ov.get("kind") == "wms":
        folium.WmsTileLayer(
            url=ov["url"],
            layers=ov["layers"],
            name=overlay_name,
            attr=ov["attr"],
            fmt=ov.get("fmt", "image/png"),
            transparent=ov.get("transparent", True),
            overlay=True,
            control=True,
            opacity=opacity,
        ).add_to(m)
    else:
        folium.TileLayer(
            tiles=ov["tiles"],
            attr=ov["attr"],
            name=overlay_name,
            overlay=True,
            control=True,
            opacity=opacity,
        ).add_to(m)
def add_categorical_legend(m: folium.Map, title: str, color_map: dict[str, str]) -> None:
    rows = "".join(
        f"""
        <div style="display:flex;align-items:center;gap:8px;margin:3px 0;">
            <span style="display:inline-block;width:12px;height:12px;border-radius:50%;
                         background:{c};border:1px solid rgba(0,0,0,0.15);"></span>
            <span style="font-size:12px;">{k}</span>
        </div>
        """
        for k, c in color_map.items()
    )
    html = f"""
    <div style="position:fixed; bottom:34px; left:42px; z-index:9999;
                background:rgba(255,255,255,0.95); border:1px solid #d5c9b7;
                border-radius:8px; padding:10px 12px; min-width:140px;
                box-shadow:0 2px 8px rgba(0,0,0,0.15);">
        <div style="font-size:11px; font-weight:700; margin-bottom:6px;">{title}</div>
        {rows}
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))


def add_numeric_legend(m: folium.Map, title: str, vmin: float, vmax: float) -> None:
    html = f"""
    <div style="position:fixed; bottom:34px; left:42px; z-index:9999;
                background:rgba(255,255,255,0.95); border:1px solid #d5c9b7;
                border-radius:8px; padding:10px 12px; min-width:160px;
                box-shadow:0 2px 8px rgba(0,0,0,0.15);">
        <div style="font-size:11px; font-weight:700; margin-bottom:6px;">{title}</div>
        <div style="width:140px;height:12px;border-radius:4px;
                    background:linear-gradient(to right,#313695,#74add1,#ffffbf,#f46d43,#a50026);
                    border:1px solid #ccc; margin-bottom:4px;"></div>
        <div style="display:flex;justify-content:space-between;font-size:10px;">
            <span>{vmin:.2f}</span><span>{(vmin+vmax)/2:.2f}</span><span>{vmax:.2f}</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))


def numeric_color(value: float, vmin: float, vmax: float) -> str:
    if pd.isna(value):
        return "#9ca3af"
    x = (value - vmin) / (vmax - vmin + 1e-9)
    x = min(max(x, 0.0), 1.0)
    r = int(49 + x * 150)
    g = int(57 + (1 - x) * 130)
    b = int(120 - x * 40)
    return f"#{r:02x}{g:02x}{b:02x}"


def ensure_color_map(labels: list[str]) -> dict[str, str]:
    cmap = dict(BASE_COLOR_MAP)
    palette = (
        px.colors.qualitative.Safe
        + px.colors.qualitative.Set3
        + px.colors.qualitative.Pastel
        + px.colors.qualitative.Vivid
    )
    unseen = [x for x in sorted(pd.Series(labels).dropna().astype(str).unique().tolist()) if x not in cmap]
    for i, lab in enumerate(unseen):
        cmap[lab] = palette[i % len(palette)]
    return cmap


def germany_gdf_3857() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame.from_features([GERMANY_GEOJSON], crs="EPSG:4326").to_crs(epsg=3857)


def plotly_to_png_bytes(fig) -> bytes | None:
    if not HAS_KALEIDO:
        return None
    buf = io.BytesIO()
    fig.write_image(buf, format="png", scale=2)
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════════════
# REAL DATA LOADING
# ══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def make_demo_data(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lats = rng.uniform(47.3, 50.5, n)
    lons = rng.uniform(8.5, 13.8, n)
    lulc = rng.choice(["Forest", "Cropland", "Urban", "Wetland", "Grassland"], n, p=[0.32, 0.33, 0.13, 0.08, 0.14])
    canopy = np.where(
        lulc == "Forest",
        rng.normal(22, 5, n),
        np.where(lulc == "Urban", rng.normal(9, 3, n), rng.normal(2, 1.5, n)),
    ).clip(0, 45)
    cloud_frac = (0.35 + 0.3 * (canopy / 45) + rng.normal(0, 0.10, n)).clip(0, 1)
    ndvi = np.where(
        lulc == "Forest",
        rng.uniform(0.62, 0.90, n),
        np.where(lulc == "Cropland", rng.uniform(0.28, 0.72, n), np.where(lulc == "Urban", rng.uniform(0.04, 0.24, n), rng.uniform(0.20, 0.62, n))),
    )
    cth = (canopy * 60 + rng.normal(0, 300, n)).clip(200, 8000)
    dates = pd.to_datetime("2019-05-26") + pd.to_timedelta(rng.integers(0, 98, n), unit="D")
    month_lbl = {5: "May", 6: "Jun", 7: "Jul", 8: "Aug"}
    return pd.DataFrame({
        "lat": lats.round(4),
        "lon": lons.round(4),
        "lulc": lulc,
        "canopy_height_m": canopy.round(2),
        "cloud_fraction": cloud_frac.round(3),
        "ndvi": ndvi.round(3),
        "cloud_top_height_m": cth.round(0),
        "date": dates,
        "month": pd.to_datetime(dates).month.map(month_lbl),
        "source": "synthetic",
    })


def _normalise_project_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for lat_c in ("gedi_lat", "modis_lat", "latitude", "lat"):
        if lat_c in out.columns:
            out.rename(columns={lat_c: "lat"}, inplace=True)
            break

    for lon_c in ("gedi_lon", "modis_lon", "longitude", "lon"):
        if lon_c in out.columns:
            out.rename(columns={lon_c: "lon"}, inplace=True)
            break

    for lc_c in (
        "lulc",
        "corine_label",
        "corine_class",
        "landcover",
        "landcover_label",
        "CLC_LABEL",
        "LABEL3",
        "label3",
    ):
        if lc_c in out.columns and "lulc" not in out.columns:
            out.rename(columns={lc_c: "lulc"}, inplace=True)
            break

    for st_c in ("state", "region", "bundesland"):
        if st_c in out.columns and "state" not in out.columns:
            out.rename(columns={st_c: "state"}, inplace=True)
            break

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    elif "source_file" in out.columns:
        out["date"] = pd.to_datetime(
            out["source_file"].astype(str).str.extract(r"(\d{4}-\d{2}-\d{2})")[0],
            errors="coerce"
        )

    if "rh98" in out.columns and "canopy_height_m" not in out.columns:
        out["canopy_height_m"] = pd.to_numeric(out["rh98"], errors="coerce").clip(0, 60)

    if "Cloud_Fraction_Day" in out.columns and "cloud_fraction" not in out.columns:
        cf = pd.to_numeric(out["Cloud_Fraction_Day"], errors="coerce")
        if cf.dropna().max() > 1.5:
            cf = cf / 100.0
        out["cloud_fraction"] = cf.clip(0, 1)

    if "Cloud_Top_Height" in out.columns and "cloud_top_height_m" not in out.columns:
        cth = pd.to_numeric(out["Cloud_Top_Height"], errors="coerce")
        out["cloud_top_height_m"] = cth.where(cth > 0, np.nan)

    if "Sur_Reflectance_NIR" in out.columns and "Sur_Reflectance_Red" in out.columns:
        nir = pd.to_numeric(out["Sur_Reflectance_NIR"], errors="coerce")
        red = pd.to_numeric(out["Sur_Reflectance_Red"], errors="coerce")
        out["ndvi"] = ((nir - red) / (nir + red + 1e-9)).clip(-1, 1)

    if "Surface_Temperature" in out.columns:
        st_val = pd.to_numeric(out["Surface_Temperature"], errors="coerce")
        out["Surface_Temperature"] = st_val.where(st_val > -1000, np.nan)

    if "Cloud_Optical_Thickness" in out.columns:
        out["Cloud_Optical_Thickness"] = pd.to_numeric(out["Cloud_Optical_Thickness"], errors="coerce").clip(0, None)

    if "month" not in out.columns:
        month_lbl = {5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct"}
        out["month"] = pd.to_datetime(out["date"], errors="coerce").dt.month.map(month_lbl).fillna("Other")
    else:
        out["month"] = out["month"].astype(str).str.split().str[0]

    if "retrieval_qc_ok" in out.columns:
        out = out[out["retrieval_qc_ok"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()

    out = out.dropna(subset=["lat", "lon"])
    out = out[out["lat"].between(47.0, 55.5) & out["lon"].between(5.5, 15.5)]

    if "lulc" not in out.columns:
        out["lulc"] = "Unknown"

    out["source"] = "project_data"
    return out.reset_index(drop=True)


@st.cache_data(show_spinner="Loading project CSV files…")
def load_project_data(data_dir: str, max_rows_per_file: int = 5000) -> pd.DataFrame:
    base = Path(data_dir)
    csvs = sorted([p for p in base.glob("*.csv") if len(p.stem) == 10 and p.stem[4] == "-"])
    if not csvs:
        return pd.DataFrame()

    frames = []
    for csv_path in csvs:
        try:
            chunk = pd.read_csv(csv_path, nrows=max_rows_per_file)
            chunk["source_file"] = csv_path.name
            frames.append(chunk)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    return _normalise_project_df(combined)


@st.cache_data(show_spinner="Fetching DWD station data…", ttl=3600)
def load_dwd_stations() -> pd.DataFrame:
    url = (
        "https://opendata.dwd.de/climate_environment/CDC/observations_germany/"
        "climate/annual/kl/recent/KL_Jahreswerte_Beschreibung_Stationen.txt"
    )
    try:
        resp = requests.get(url, timeout=12)
        resp.raise_for_status()
        lines = resp.content.decode("latin-1").splitlines()

        records = []
        for line in lines[2:]:
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                records.append({
                    "station_id": parts[0],
                    "elevation_m": float(parts[3]),
                    "lat": float(parts[4]),
                    "lon": float(parts[5]),
                    "state": parts[-1],
                })
            except Exception:
                continue

        df = pd.DataFrame(records)
        if df.empty:
            return pd.DataFrame()

        df = df[df["lat"].between(47.2, 55.1) & df["lon"].between(5.8, 15.1)].copy()
        rng = np.random.default_rng(42)
        n = len(df)
        df["cloud_fraction"] = (0.45 + rng.normal(0, 0.08, n)).clip(0.1, 0.95)
        df["ndvi"] = (0.55 + rng.normal(0, 0.10, n)).clip(0.05, 0.92)
        df["canopy_height_m"] = (rng.exponential(8, n) + 1).clip(0, 40)
        df["cloud_top_height_m"] = (df["cloud_fraction"] * 6000 + rng.normal(0, 400, n)).clip(200, 9000)
        df["lulc"] = rng.choice(["Forest", "Cropland", "Urban", "Wetland", "Grassland"], n, p=[0.28, 0.35, 0.14, 0.08, 0.15])
        df["month"] = rng.choice(["May", "Jun", "Jul", "Aug"], n)
        df["date"] = pd.to_datetime("2019-05-01") + pd.to_timedelta(rng.integers(0, 98, n), unit="D")
        df["source"] = "DWD"
        return df.reset_index(drop=True)

    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
# EARTH ENGINE: REAL SENTINEL-2 RGB
# ══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def init_ee():
    if not HAS_EE:
        return False
    try:
        ee.Initialize()
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize()
        except Exception:
            return False
    return True


def germany_geom_ee():
    return ee.Geometry.Polygon(GERMANY_GEOJSON["geometry"]["coordinates"])


def available_story_dates(df_in: pd.DataFrame) -> list[str]:
    if "date" not in df_in.columns:
        return []
    d = pd.to_datetime(df_in["date"], errors="coerce").dropna()
    if d.empty:
        return []
    return sorted(d.dt.normalize().dt.strftime("%Y-%m-%d").unique().tolist())


@st.cache_data(show_spinner=False)
def get_s2_rgb_overlay(selected_date: str, window_days: int = 0):
    if not init_ee():
        return None

    start = pd.Timestamp(selected_date).normalize()
    end = start + pd.Timedelta(days=int(window_days) + 1)
    geom = germany_geom_ee()

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geom)
        .filterDate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    )

    count = collection.size().getInfo()
    if count == 0:
        return None

    image = collection.mosaic().clip(geom)
    vis = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000, "gamma": 1.1}
    map_id = image.getMapId(vis)

    if window_days == 0:
        label = f"Sentinel-2 RGB | {start.strftime('%Y-%m-%d')} | unfiltered"
    else:
        end_label = (end - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        label = f"Sentinel-2 RGB | {start.strftime('%Y-%m-%d')} to {end_label} | unfiltered"

    return {
        "tiles": map_id["tile_fetcher"].url_format,
        "attr": "Copernicus Sentinel-2 via Google Earth Engine",
        "name": label,
        "count": int(count),
    }


def add_s2_overlay(m: folium.Map, s2_overlay: dict | None, opacity: float) -> None:
    if s2_overlay:
        folium.TileLayer(
            tiles=s2_overlay["tiles"],
            attr=s2_overlay["attr"],
            name=s2_overlay["name"],
            overlay=True,
            control=True,
            opacity=opacity,
        ).add_to(m)


# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════
project_raw = load_project_data(str(DATA_DIR))
dwd_raw = load_dwd_stations()
synthetic_raw = make_demo_data()

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛰️ Satellite Basemaps")
    st.markdown("### *for Science Storytelling*")
    st.markdown("---")

    source_options = []
    if not project_raw.empty:
        source_options.append("📂 Project data (MODIS+GEDI+CORINE)")
    if not dwd_raw.empty:
        source_options.append("🌍 DWD stations (open)")
    source_options.append("🔬 Synthetic demo")

    data_src = st.radio("Data source", source_options, index=0)

    if data_src.startswith("📂"):
        df_all = project_raw.copy()
    elif data_src.startswith("🌍"):
        df_all = dwd_raw.copy()
    else:
        df_all = synthetic_raw.copy()

    if "lulc" not in df_all.columns:
        df_all["lulc"] = "Unknown"

    color_map = ensure_color_map(df_all["lulc"].astype(str).tolist())
    numeric_vars = [c for c in REAL_NUMERIC_CANDIDATES if c in df_all.columns and pd.api.types.is_numeric_dtype(df_all[c])]
    if not numeric_vars:
        numeric_vars = ["cloud_fraction", "canopy_height_m", "ndvi", "cloud_top_height_m"]
        numeric_vars = [c for c in numeric_vars if c in df_all.columns]

    st.markdown("**Filters**")
    lulc_options = sorted(df_all["lulc"].dropna().astype(str).unique().tolist())
    month_options = sorted(df_all["month"].dropna().astype(str).unique().tolist()) if "month" in df_all.columns else []

    sel_lulc = st.multiselect(
        "Land cover classes",
        lulc_options,
        default=lulc_options[: min(6, len(lulc_options))],
    )
    sel_months = st.multiselect(
        "Months",
        month_options,
        default=month_options,
    )

    max_points = st.slider("Max points on map", 500, 10000, 3000, 500)

    st.markdown("---")
    base_tile = st.selectbox("Default basemap", list(TILE_LAYERS.keys()), index=3)
    plotly_tile = st.selectbox("Plotly basemap", list(PLOTLY_BASEMAPS.keys()), index=0)

    st.markdown("---")
    st.markdown("**Real Sentinel-2 RGB**")

    date_source = project_raw.copy() if not project_raw.empty else df_all.copy()
    s2_date_options = available_story_dates(date_source)

    s2_enabled = False
    s2_overlay = None
    s2_opacity = 0.90

    if HAS_EE and s2_date_options:
        s2_enabled = st.checkbox(
            "Show real Sentinel-2 RGB",
            value=data_src.startswith("📂"),
            help="Real Copernicus Sentinel-2 RGB via Earth Engine. No cloud filtering.",
        )
        s2_date = st.select_slider("Sentinel-2 date", options=s2_date_options, value=s2_date_options[0])
        s2_window_days = st.slider("Context window (days)", 0, 6, 0)
        s2_opacity = st.slider("Sentinel-2 opacity", 0.1, 1.0, 0.90, 0.05)
        if s2_enabled:
            s2_overlay = get_s2_rgb_overlay(s2_date, s2_window_days)
            if s2_overlay:
                st.caption(f"{s2_overlay['name']} · {s2_overlay['count']} image(s)")
            else:
                st.caption("No Sentinel-2 image found for that day/window.")
    elif not HAS_EE:
        st.caption("Install/authenticate Earth Engine to use real Sentinel-2.")
    else:
        st.caption("No usable dates found in current data.")

    with st.expander("Debug: loaded data"):
        st.write("Rows:", len(df_all))
        st.write("Columns:", list(df_all.columns))
        if "date" in df_all.columns:
            st.write("Date range:", pd.to_datetime(df_all["date"], errors="coerce").min(), "→", pd.to_datetime(df_all["date"], errors="coerce").max())
        st.write("LULC sample:", lulc_options[:10])

# Apply filters
df = df_all.copy()
if sel_lulc:
    df = df[df["lulc"].astype(str).isin(sel_lulc)]
if sel_months and "month" in df.columns:
    df = df[df["month"].astype(str).isin(sel_months)]
if len(df) > max_points:
    df = df.sample(max_points, random_state=42).copy()

# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown(
    """
<div style="border-bottom:2px solid #d5c9b7; padding-bottom:18px; margin-bottom:10px;">
  <div style="font-size:0.72rem; letter-spacing:0.12em; text-transform:uppercase; color:#166534; margin-bottom:4px;">
   LIM Weekly talks · Leipzig University / ScaDS.AI
  </div>
  <h1 style="margin:0 0 8px;">Satellite Basemaps for Science Storytelling</h1>
  <div style="color:#4b5563; max-width:850px;">
    Using real satellite imagery as visual context for MODIS, GEDI, and CORINE-derived information,
    with interactive maps, static figures, downloads, and storytelling panels.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if s2_enabled and s2_overlay:
    st.markdown(f'<div class="story-badge">🛰️ {s2_overlay["name"]}</div>', unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Points on map", f"{len(df):,}")
m2.metric("Total loaded", f"{len(df_all):,}")
m3.metric("Mean cloud fraction", f"{df['cloud_fraction'].mean():.3f}" if "cloud_fraction" in df.columns and df["cloud_fraction"].notna().any() else "—")
m4.metric("Mean NDVI", f"{df['ndvi'].mean():.3f}" if "ndvi" in df.columns and df["ndvi"].notna().any() else "—")
m5.metric("Data source", "Project" if data_src.startswith("📂") else "DWD" if data_src.startswith("🌍") else "Synthetic")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
(
    tab_why,
    tab_tiles,
    tab_folium,
    tab_draw,
    tab_static,
    tab_plotly,
    tab_download,
    tab_story,
    tab_ref,
) = st.tabs([
    "🧭 Why Basemaps",
    "🛰️ Tile Services",
    "🗺️ Folium / Interactive",
    "✏️ Draw & Annotate",
    "🖼️ Static Maps",
    "📊 Plotly",
    "⬇️ Download for Talks",
    "📖 Storytelling",
    "📚 Reference",
])

# ══════════════════════════════════════════════════════════════════
# TAB: WHY
# ══════════════════════════════════════════════════════════════════
with tab_why:
    st.markdown('<div class="ch-pill">Chapter 0</div>', unsafe_allow_html=True)
    st.markdown("## Why satellite basemaps matter")
    st.markdown(
        """
<div class="k-card">
<b>Satellite basemaps are not just background imagery.</b> They provide environmental context that helps
an audience understand where a pattern occurs, what surrounds it, and why it matters. They are strongest
when your story depends on visible landscape context: forests, croplands, cities, wetlands, or terrain.
</div>
<div class="k-card">
<b>Good use cases</b><br>
• cloud, vegetation, or land-cover stories<br>
• explaining spatial context to non-GIS audiences<br>
• comparing regions or time slices with visible real-world background<br>
• presentation storytelling where location matters visually
</div>
<div class="k-card">
<b>Be careful</b><br>
• a strong basemap can overpower your thematic layer<br>
• date mismatch matters<br>
• mixed scales matter: Sentinel-2, MODIS, GEDI, and CORINE are not the same resolution
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("### The basemap comparison — same data, different canvas")
    st.markdown("""
<table class="cmp-table">
<tr>
  <th>Basemap</th><th>Best for</th><th>Free?</th><th>Print quality</th><th>Package</th>
</tr>
<tr>
  <td>🛰️ Esri Satellite</td>
  <td>Vegetation, terrain, LULC stories</td>
  <td class="yes">Yes (attr)</td><td class="partial">Medium</td><td>folium, contextily, leafmap</td>
</tr>
<tr>
  <td>🌿 Sentinel-2 RGB</td>
  <td>High-res recent imagery, Europe</td>
  <td class="yes">Yes (EOX tile)</td><td class="yes">High</td><td>folium, leafmap, geemap</td>
</tr>
<tr>
  <td>📡 MODIS True Color</td>
  <td>Daily cloud/atmosphere overlays</td>
  <td class="yes">Yes (NASA GIBS)</td><td class="partial">Medium</td><td>folium, leafmap</td>
</tr>
<tr>
  <td>🗺️ CartoDB Positron</td>
  <td>Light clean base, data-forward maps</td>
  <td class="yes">Yes</td><td class="yes">High</td><td>all packages</td>
</tr>
<tr>
  <td>🌙 CartoDB Dark</td>
  <td>Bright data on dark canvas, night-sky feel</td>
  <td class="yes">Yes</td><td class="yes">High</td><td>all packages</td>
</tr>
<tr>
  <td>🗻 Esri Topo / NatGeo</td>
  <td>Terrain-aware stories, mountainous areas</td>
  <td class="yes">Yes (attr)</td><td class="yes">High</td><td>folium, contextily</td>
</tr>
<tr>
  <td>🌍 Google Satellite</td>
  <td>Highest resolution global imagery</td>
  <td class="partial">API key</td><td class="yes">Very high</td><td>geemap, leafmap</td>
</tr>
</table>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB: TILE SERVICES
# ══════════════════════════════════════════════════════════════════
with tab_tiles:
    st.markdown('<div class="ch-pill">Chapter 1</div>', unsafe_allow_html=True)
    st.markdown("## Live tile service explorer")

    c1, c2 = st.columns([1, 2.4])
    with c1:
        tile_choice = st.selectbox("Basemap", list(TILE_LAYERS.keys()), index=list(TILE_LAYERS.keys()).index(base_tile))
        active_overlays = []
        for ov in RS_OVERLAYS:
            if st.checkbox(ov, value=(ov == "Sentinel-2 Cloudless 2021 (EOX)"), key=f"tile_{ov}"):
                active_overlays.append(ov)
        ov_opacity = st.slider("Overlay opacity", 0.1, 1.0, 0.9, 0.05, key="tiles_op")
        show_boundary = st.checkbox("Show Germany boundary", True, key="tiles_boundary")

    with c2:
        key = hashlib.md5(f"{tile_choice}|{active_overlays}|{ov_opacity}|{show_boundary}".encode()).hexdigest()[:8]
        m = make_folium_map(tile_choice)
        
        for ov in active_overlays:
            add_s2_overlay(m, s2_overlay if s2_enabled else None, s2_opacity)
            add_overlay_to_map(m, ov, ov_opacity)
        if show_boundary:
            add_germany_boundary(m)
        folium.LayerControl(collapsed=False).add_to(m)
        st_folium(m, width="100%", height=580, key=f"tile_explorer_{key}")

# ══════════════════════════════════════════════════════════════════
            #     overlay=True,
            #     control=True,
            #     opacity=ov_opacity,
            # ).add_to(m)
        if show_boundary:
            add_germany_boundary(m)
        folium.LayerControl(collapsed=False).add_to(m)
        st_folium(m, width="100%", height=580, key=f"tile_explorer_{key}")

# ══════════════════════════════════════════════════════════════════
# TAB: FOLIUM / INTERACTIVE
# ══════════════════════════════════════════════════════════════════
with tab_folium:
    st.markdown('<div class="ch-pill">Chapter 2</div>', unsafe_allow_html=True)
    st.markdown("## Folium / Interactive")
    st.markdown("### MODIS + CORINE + GEDI summaries")

    csum1, csum2 = st.columns(2)
    with csum1:
        if "cloud_fraction" in df.columns and "lulc" in df.columns:
            top_lc = df["lulc"].astype(str).value_counts().head(8).index.tolist()
            sub = df[df["lulc"].astype(str).isin(top_lc)].dropna(subset=["cloud_fraction"])
            if not sub.empty:
                fig_box = px.box(
                    sub,
                    x="lulc",
                    y="cloud_fraction",
                    color="lulc",
                    color_discrete_map=color_map,
                    title="Cloud fraction by CORINE land cover",
                    labels={"cloud_fraction": "Cloud fraction", "lulc": "Land cover"},
                    height=340,
                )
                fig_box.update_layout(showlegend=False, xaxis_tickangle=-30, margin={"t": 50, "b": 70})
                st.plotly_chart(fig_box, use_container_width=True)

    with csum2:
        if "canopy_height_m" in df.columns and "cloud_fraction" in df.columns and "lulc" in df.columns:
            sub = df.dropna(subset=["canopy_height_m", "cloud_fraction", "lulc"]).copy()
            if len(sub) > 2000:
                sub = sub.sample(2000, random_state=1)
            if not sub.empty:
                fig_sc = px.scatter(
                    sub,
                    x="canopy_height_m",
                    y="cloud_fraction",
                    color="lulc",
                    color_discrete_map=color_map,
                    trendline="ols",
                    trendline_scope="overall",
                    opacity=0.6,
                    height=340,
                    title="GEDI canopy height → MODIS cloud fraction",
                )
                fig_sc.update_layout(showlegend=False, margin={"t": 50})
                st.plotly_chart(fig_sc, use_container_width=True)

    if "ndvi" in df.columns and "lulc" in df.columns:
        top_lc = df["lulc"].astype(str).value_counts().head(8).index.tolist()
        sub = df[df["lulc"].astype(str).isin(top_lc)].dropna(subset=["ndvi"])
        if not sub.empty:
            fig_vio = px.violin(
                sub,
                x="lulc",
                y="ndvi",
                color="lulc",
                color_discrete_map=color_map,
                box=True,
                title="NDVI by CORINE land cover",
                height=340,
            )
            fig_vio.update_layout(showlegend=False, xaxis_tickangle=-30, margin={"t": 50, "b": 70})
            st.plotly_chart(fig_vio, use_container_width=True)

    st.markdown("---")
    st.markdown("### Interactive map")

    cf1, cf2 = st.columns([1, 2.5])
    # with cf1:
    #     layer_mode = st.radio("Layer type", ["Circle Markers", "Clustered", "Heatmap", "Category Groups"])
    #     color_var = st.selectbox("Map variable", numeric_vars, index=0)
    #     show_minimap = st.checkbox("MiniMap", True)
    #     show_measure = st.checkbox("Measure tool", True)
    #     show_boundary = st.checkbox("Show Germany boundary", True)
    with cf1:
        layer_mode = st.radio("Layer type", ["Circle Markers", "Clustered", "Heatmap", "Category Groups"])
        color_var = st.selectbox("Map variable", numeric_vars, index=0)
        show_minimap = st.checkbox("MiniMap", True)
        show_measure = st.checkbox("Measure tool", True)
        show_boundary = st.checkbox("Show Germany boundary", True)

        show_esa_class = st.checkbox("Show ESA WorldCover classification", True)
        show_esa_rgb = st.checkbox("Show ESA WorldCover RGB", False)
        esa_opacity = st.slider("ESA layer opacity", 0.1, 1.0, 0.65, 0.05)
    with cf2:
        key = hashlib.md5(f"{base_tile}|{layer_mode}|{color_var}|{show_boundary}|{s2_overlay['name'] if s2_overlay else 'none'}".encode()).hexdigest()[:8]
        m = make_folium_map(base_tile)
        add_s2_overlay(m, s2_overlay if s2_enabled else None, s2_opacity)
        add_s2_overlay(m, s2_overlay if s2_enabled else None, s2_opacity)

        if show_esa_rgb:
            add_overlay_to_map(m, "ESA WorldCover 2021 RGB", opacity=esa_opacity)

        if show_esa_class:
            add_overlay_to_map(m, "ESA WorldCover 2021 Classification", opacity=esa_opacity)

        if show_minimap:
            MiniMap(toggle_display=True).add_to(m)
        if show_measure:
            MeasureControl(position="bottomright").add_to(m)
        if show_boundary:
            add_germany_boundary(m)
        if show_esa_class:
            add_categorical_legend(m, "ESA WorldCover 2021", ESA_WORLDCOVER_LEGEND)
        if layer_mode == "Circle Markers":
            vals = pd.to_numeric(df[color_var], errors="coerce")
            vmin, vmax = float(vals.min()), float(vals.max())
            for _, row in df.dropna(subset=["lat", "lon", color_var]).iterrows():
                c = numeric_color(float(row[color_var]), vmin, vmax)
                folium.CircleMarker(
                    [row["lat"], row["lon"]],
                    radius=6,
                    color=c,
                    fill=True,
                    fill_opacity=0.82,
                    weight=1,
                    tooltip=f"{row.get('lulc','Unknown')} · {color_var}: {row[color_var]:.3f}",
                ).add_to(m)
            add_numeric_legend(m, color_var, vmin, vmax)

        elif layer_mode == "Clustered":
            cluster = MarkerCluster(name="Observations").add_to(m)
            for _, row in df.dropna(subset=["lat", "lon"]).iterrows():
                lc = str(row.get("lulc", "Unknown"))
                folium.Marker(
                    [row["lat"], row["lon"]],
                    tooltip=f"{lc} · {color_var}: {row.get(color_var, np.nan)}",
                    icon=folium.Icon(color="green", icon="leaf", prefix="fa"),
                ).add_to(cluster)
            visible = {k: v for k, v in color_map.items() if k in df["lulc"].astype(str).unique().tolist()}
            add_categorical_legend(m, "Land Cover", visible)

        elif layer_mode == "Heatmap":
            vals = pd.to_numeric(df[color_var], errors="coerce")
            vmin, vmax = float(vals.min()), float(vals.max())
            norm = ((vals - vmin) / (vmax - vmin + 1e-9)).fillna(0)
            heat_data = list(zip(df["lat"].tolist(), df["lon"].tolist(), norm.tolist()))
            HeatMap(
                heat_data,
                radius=20,
                blur=14,
                min_opacity=0.35,
                gradient={0.2: "#313695", 0.5: "#ffffbf", 0.85: "#a50026"},
            ).add_to(m)
            add_numeric_legend(m, f"{color_var} intensity", vmin, vmax)

        else:
            groups = {}
            for _, row in df.dropna(subset=["lat", "lon"]).iterrows():
                lc = str(row.get("lulc", "Unknown"))
                groups.setdefault(lc, folium.FeatureGroup(name=f"● {lc}", show=True))
                folium.CircleMarker(
                    [row["lat"], row["lon"]],
                    radius=5,
                    color=color_map.get(lc, "#9ca3af"),
                    fill=True,
                    fill_opacity=0.78,
                    weight=1,
                    tooltip=f"{lc} · {color_var}: {row.get(color_var, np.nan)}",
                ).add_to(groups[lc])
            for fg in groups.values():
                fg.add_to(m)
            visible = {k: v for k, v in color_map.items() if k in df["lulc"].astype(str).unique().tolist()}
            add_categorical_legend(m, "Land Cover", visible)

        folium.LayerControl(collapsed=False).add_to(m)
        st_folium(m, width="100%", height=620, key=f"folium_main_{key}")

# ══════════════════════════════════════════════════════════════════
# TAB: DRAW
# ══════════════════════════════════════════════════════════════════
with tab_draw:
    st.markdown('<div class="ch-pill">Chapter 3</div>', unsafe_allow_html=True)
    st.markdown("## Draw & Annotate")
    st.markdown(
        """
<div class="draw-info">
Draw a polygon or rectangle over Germany. Points inside it will be selected and summarised below.
</div>
""",
        unsafe_allow_html=True,
    )

    m = make_folium_map(base_tile)
    add_s2_overlay(m, s2_overlay if s2_enabled else None, s2_opacity)
    add_germany_boundary(m)
    Draw(
        export=True,
        draw_options={
            "polygon": {"allowIntersection": False},
            "rectangle": True,
            "circle": True,
            "polyline": False,
            "marker": False,
            "circlemarker": False,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    for _, row in df.dropna(subset=["lat", "lon"]).iterrows():
        folium.CircleMarker(
            [row["lat"], row["lon"]],
            radius=5,
            color=color_map.get(str(row.get("lulc", "Unknown")), "#9ca3af"),
            fill=True,
            fill_opacity=0.72,
            weight=1,
            tooltip=f"{row.get('lulc','Unknown')} · cloud: {row.get('cloud_fraction', np.nan):.3f}",
        ).add_to(m)

    visible = {k: v for k, v in color_map.items() if k in df["lulc"].astype(str).unique().tolist()}
    add_categorical_legend(m, "Land Cover", visible)
    folium.LayerControl().add_to(m)

    draw_result = st_folium(m, width="100%", height=600, key="draw_map", returned_objects=["all_drawings"])

    drawings = draw_result.get("all_drawings") or []
    df_sel = pd.DataFrame()

    if drawings:
        masks = []
        for d in drawings:
            try:
                geom = shapely_shape(d["geometry"])
                masks.append(df.apply(lambda r: geom.contains(Point(r["lon"], r["lat"])), axis=1))
            except Exception:
                continue
        if masks:
            combined = masks[0]
            for mk in masks[1:]:
                combined = combined | mk
            df_sel = df[combined].copy()

    if not df_sel.empty:
        st.success(f"{len(df_sel)} points selected")
        a, b, c, d = st.columns(4)
        a.metric("Canopy", f"{df_sel['canopy_height_m'].mean():.1f} m" if "canopy_height_m" in df_sel.columns else "—")
        b.metric("Cloud fraction", f"{df_sel['cloud_fraction'].mean():.3f}" if "cloud_fraction" in df_sel.columns else "—")
        c.metric("NDVI", f"{df_sel['ndvi'].mean():.3f}" if "ndvi" in df_sel.columns else "—")
        d.metric("N points", len(df_sel))

        st.download_button(
            "Download selected points (CSV)",
            df_sel.to_csv(index=False).encode("utf-8"),
            "selected_points.csv",
            "text/csv",
        )
    elif drawings:
        st.warning("No points inside drawn geometry.")

# ══════════════════════════════════════════════════════════════════
# TAB: STATIC
# ══════════════════════════════════════════════════════════════════
with tab_static:
    st.markdown('<div class="ch-pill">Chapter 4</div>', unsafe_allow_html=True)
    st.markdown("## Static maps")

    if not HAS_CTX:
        st.warning("Install contextily and matplotlib to use this section.")
    else:
        c1, c2 = st.columns([1, 2.5])
        with c1:
            ctx_provider_name = st.selectbox(
                "contextily provider",
                ["Esri.WorldImagery", "CartoDB.Positron", "CartoDB.DarkMatter"],
                index=0,
            )
            static_var = st.selectbox("Color variable", numeric_vars, index=0, key="static_var")
            static_cmap = st.selectbox("Colormap", ["RdYlBu_r", "viridis", "plasma", "YlGnBu", "Greens"], index=0)
            marker_size = st.slider("Marker size", 10, 80, 35, 5)
            basemap_alpha = st.slider("Basemap alpha", 0.3, 1.0, 0.82, 0.05)
            dpi_choice = st.select_slider("DPI", [72, 100, 150, 200, 300], value=150)
            fig_title = st.text_input("Figure title", "Germany AOI + point observations")

        with c2:
            gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326").to_crs(epsg=3857)
            germany_web = germany_gdf_3857()

            fig, ax = plt.subplots(figsize=(10, 8), dpi=100, facecolor="#f7f4ee")
            ax.set_facecolor("#f7f4ee")

            gdf.plot(
                ax=ax,
                column=static_var,
                cmap=static_cmap,
                markersize=marker_size,
                alpha=0.88,
                legend=True,
                legend_kwds={"label": static_var, "shrink": 0.55},
            )

            provider_obj = ctx.providers
            for part in ctx_provider_name.split("."):
                provider_obj = getattr(provider_obj, part)
            ctx.add_basemap(ax, source=provider_obj, crs="EPSG:3857", alpha=basemap_alpha)

            # Germany AOI
            germany_web.boundary.plot(ax=ax, color="#e63946", linewidth=1.8, alpha=0.95)

            ax.set_axis_off()
            ax.set_title(fig_title, fontsize=13, pad=12)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi_choice, bbox_inches="tight", facecolor="#f7f4ee")
            buf.seek(0)
            st.download_button(
                f"Download PNG ({dpi_choice} dpi)",
                buf.getvalue(),
                f"static_map_{static_var}_{dpi_choice}dpi.png",
                "image/png",
            )
            plt.close(fig)

# ══════════════════════════════════════════════════════════════════
# TAB: PLOTLY
# ══════════════════════════════════════════════════════════════════
with tab_plotly:
    st.markdown('<div class="ch-pill">Chapter 5</div>', unsafe_allow_html=True)
    st.markdown("## Plotly maps")

    p1, p2 = st.columns(2)
    with p1:
        color_col = st.selectbox("Color", ["lulc"] + numeric_vars, index=0)
        size_col = st.selectbox("Size", numeric_vars, index=0)
        animate_month = st.checkbox("Animate by month", value=False)

        fig_sc = px.scatter_mapbox(
            df,
            lat="lat",
            lon="lon",
            color=color_col,
            size=size_col,
            size_max=16,
            animation_frame="month" if animate_month and "month" in df.columns else None,
            mapbox_style=PLOTLY_BASEMAPS[plotly_tile],
            zoom=5,
            center={"lat": GERMANY_CENTER[0], "lon": GERMANY_CENTER[1]},
            color_discrete_map=color_map,
            color_continuous_scale="Viridis",
            height=500,
        )
        fig_sc.update_layout(margin={"r": 0, "t": 20, "l": 0, "b": 0})
        st.plotly_chart(fig_sc, use_container_width=True)

    with p2:
        density_var = st.selectbox("Density variable", numeric_vars, index=0, key="density_var")
        fig_den = px.density_mapbox(
            df,
            lat="lat",
            lon="lon",
            z=density_var,
            radius=28,
            mapbox_style=PLOTLY_BASEMAPS[plotly_tile],
            zoom=5,
            center={"lat": GERMANY_CENTER[0], "lon": GERMANY_CENTER[1]},
            color_continuous_scale="Turbo",
            height=500,
        )
        fig_den.update_layout(margin={"r": 0, "t": 20, "l": 0, "b": 0})
        st.plotly_chart(fig_den, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB: DOWNLOAD
# ══════════════════════════════════════════════════════════════════
with tab_download:
    st.markdown('<div class="ch-pill">Chapter 6</div>', unsafe_allow_html=True)
    st.markdown("## Download for talks")

    d1, d2 = st.columns([1, 2.4])
    with d1:
        dl_tile = st.selectbox("Basemap", list(TILE_LAYERS.keys()), index=3, key="dl_tile")
        dl_layer = st.selectbox("Layer", ["Circle Markers", "Heatmap", "Clusters"], index=0)
        dl_metric = st.selectbox("Metric", numeric_vars, index=0, key="dl_metric")

    with d2:
        key = hashlib.md5(f"{dl_tile}|{dl_layer}|{dl_metric}|{s2_overlay['name'] if s2_overlay else 'none'}".encode()).hexdigest()[:8]
        m = make_folium_map(dl_tile)
        add_s2_overlay(m, s2_overlay if s2_enabled else None, s2_opacity)
        add_germany_boundary(m)

        if dl_layer == "Circle Markers":
            vals = pd.to_numeric(df[dl_metric], errors="coerce")
            vmin, vmax = float(vals.min()), float(vals.max())
            for _, row in df.dropna(subset=["lat", "lon", dl_metric]).iterrows():
                c = numeric_color(float(row[dl_metric]), vmin, vmax)
                folium.CircleMarker(
                    [row["lat"], row["lon"]],
                    radius=6,
                    color=c,
                    fill=True,
                    fill_opacity=0.8,
                    weight=1,
                    tooltip=f"{row.get('lulc','Unknown')} · {dl_metric}: {row[dl_metric]:.3f}",
                ).add_to(m)
            add_numeric_legend(m, dl_metric, vmin, vmax)

        elif dl_layer == "Heatmap":
            vals = pd.to_numeric(df[dl_metric], errors="coerce")
            vmin, vmax = float(vals.min()), float(vals.max())
            norm = ((vals - vmin) / (vmax - vmin + 1e-9)).fillna(0)
            HeatMap(list(zip(df["lat"], df["lon"], norm)), radius=20, blur=14).add_to(m)
            add_numeric_legend(m, f"{dl_metric} intensity", vmin, vmax)

        else:
            cl = MarkerCluster(name="Observations").add_to(m)
            for _, row in df.dropna(subset=["lat", "lon"]).iterrows():
                folium.Marker(
                    [row["lat"], row["lon"]],
                    tooltip=f"{row.get('lulc','Unknown')}",
                    icon=folium.Icon(color="green", icon="leaf", prefix="fa"),
                ).add_to(cl)
            visible = {k: v for k, v in color_map.items() if k in df["lulc"].astype(str).unique().tolist()}
            add_categorical_legend(m, "Land Cover", visible)

        folium.LayerControl().add_to(m)
        st_folium(m, width="100%", height=500, key=f"download_preview_{key}")

        buf = io.BytesIO()
        m.save(buf, close_file=False)
        buf.seek(0)
        st.download_button(
            "Download interactive HTML",
            buf.getvalue(),
            "satellite_map_talk.html",
            "text/html",
        )

# ══════════════════════════════════════════════════════════════════
# TAB: STORY
# ══════════════════════════════════════════════════════════════════
with tab_story:
    st.markdown('<div class="ch-pill teal">Chapter 7</div>', unsafe_allow_html=True)
    st.markdown("## Map storytelling — composing the narrative")

    st.markdown("""
<div class="k-card teal">
<h4>🧠 WHAT MAKES A MAP TELL A STORY?</h4>
A map is not just a spatial view of data — it's an <b>argument</b>.
The audience should be able to extract your key finding within 10 seconds.
Every design choice — basemap, colormap, opacity, annotation — either supports
or undermines that argument.
</div>
""", unsafe_allow_html=True)

    s1, s2 = st.columns(2)
    with s1:
        st.markdown("""
<div class="k-card">
<h4>🎨 DESIGN PRINCIPLES FOR GEOSPATIAL STORYTELLING</h4>
<b>1. One map, one message</b><br>
Don't try to show cloud fraction, NDVI, canopy height, and land cover
simultaneously. Split into panels or simplify.<br><br>

<b>2. Choose the basemap to support your variable</b><br>
Showing vegetation/NDVI → use Esri Satellite (forests visible).<br>
Showing urban effects → use CartoDB Dark (cities glow).<br>
Showing climate anomaly → use CartoDB Positron (clean, neutral).<br><br>

<b>3. Contrast hierarchy</b><br>
Your data layer must be the brightest/most saturated element.
Set basemap opacity to 0.6–0.8 so it provides context without competing.<br><br>

<b>4. Annotation beats description</b><br>
Draw an arrow to the interesting feature on the map instead of
writing a paragraph in your caption. Audiences follow pointing.
</div>
""", unsafe_allow_html=True)

    with s2:
        st.markdown("""
<div class="k-card blue">
<h4>📐 CARTOGRAPHIC ELEMENTS TO ALWAYS INCLUDE</h4>
<b>Scale bar</b> — Never omit. Use <code>matplotlib_scalebar</code> for static maps.<br>
<b>North arrow</b> — For non-standard orientations; skip for standard north-up maps.<br>
<b>Inset / locator map</b> — Show where your study area sits in a wider context.<br>
<b>CRS / projection note</b> — Always state the projection in figure captions.<br><br>
<b>Data source + date</b> — Satellite data is time-specific; document it.
</div>

<div class="k-card green">
<h4>🎞️ ANIMATED MAPS FOR PRESENTATIONS</h4>
Two approaches for animated/time-series maps:<br><br>
<b>1. Plotly animation_frame</b> — Browser-native, interactive playback.<br>
<b>2. matplotlib FuncAnimation</b> — Render to MP4/GIF for embedding in PowerPoint.
</div>
""", unsafe_allow_html=True)

    st.markdown("### The cloud-fraction story — four versions of the same map")
    st.caption("Data: real DWD weather stations across Germany")

    s1, s2, s3, s4 = st.columns(4)
    base_kw = {
        "lat": "lat",
        "lon": "lon",
        "zoom": 5,
        "center": {"lat": GERMANY_CENTER[0], "lon": GERMANY_CENTER[1]},
        "height": 330,
        "hover_data": {"lulc": True, "cloud_fraction": ":.3f"} if "cloud_fraction" in df.columns else {"lulc": True},
    }

    with s1:
        st.caption("v1 — CartoDB Positron")
        fig1 = px.scatter_mapbox(
            df,
            color="cloud_fraction" if "cloud_fraction" in df.columns else "lulc",
            color_continuous_scale="Blues",
            mapbox_style="carto-positron",
            **base_kw,
        )
        fig1.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with s2:
        st.caption("v2 — OpenStreetMap")
        fig2 = px.scatter_mapbox(
            df,
            color="cloud_fraction" if "cloud_fraction" in df.columns else "lulc",
            color_continuous_scale="RdYlBu_r",
            mapbox_style="open-street-map",
            **base_kw,
        )
        fig2.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with s3:
        st.caption("v3 — White background")
        fig3 = px.scatter_mapbox(
            df,
            color="cloud_fraction" if "cloud_fraction" in df.columns else "lulc",
            color_continuous_scale="Plasma",
            mapbox_style="white-bg",
            **base_kw,
        )
        fig3.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with s4:
        st.caption("v4 — CartoDB Dark + categories")
        fig4 = px.scatter_mapbox(
            df,
            color="lulc",
            color_discrete_map=color_map,
            mapbox_style="carto-darkmatter",
            **base_kw,
        )
        fig4.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### Animated time-series — cloud fraction by day")
    if "date" in df.columns and pd.to_datetime(df["date"], errors="coerce").notna().any() and "cloud_fraction" in df.columns:
        df_anim = df.copy()
        df_anim["date"] = pd.to_datetime(df_anim["date"], errors="coerce")
        df_anim = df_anim.dropna(subset=["date"]).copy()
        df_anim["date_str"] = df_anim["date"].dt.strftime("%Y-%m-%d")
        ordered_dates = sorted(df_anim["date_str"].unique().tolist())

        fig_anim = px.scatter_mapbox(
            df_anim,
            lat="lat",
            lon="lon",
            color="cloud_fraction",
            size="canopy_height_m" if "canopy_height_m" in df_anim.columns else None,
            size_max=16,
            animation_frame="date_str",
            category_orders={"date_str": ordered_dates},
            hover_data={"lulc": True, "cloud_fraction": ":.3f", "date_str": True},
            color_continuous_scale="RdYlBu_r",
            range_color=[0, 1],
            mapbox_style="carto-positron",
            zoom=5,
            center={"lat": GERMANY_CENTER[0], "lon": GERMANY_CENTER[1]},
            height=520,
            title="Cloud fraction by day",
        )
        fig_anim.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})

        if fig_anim.layout.updatemenus:
            try:
                fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 800
                fig_anim.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 300
            except Exception:
                pass

        if fig_anim.layout.sliders:
            fig_anim.layout.sliders[0]["currentvalue"] = {"prefix": "Date: "}

        st.plotly_chart(fig_anim, use_container_width=True)

        if HAS_KALEIDO:
            png = plotly_to_png_bytes(fig_anim)
            if png:
                st.download_button("Download first frame as PNG", png, "story_animation_frame.png", "image/png")
    else:
        st.info("Animation needs valid `date` and `cloud_fraction` columns.")

# ══════════════════════════════════════════════════════════════════
# TAB: REFERENCE
# ══════════════════════════════════════════════════════════════════
with tab_ref:
    st.markdown('<div class="ch-pill">Reference</div>', unsafe_allow_html=True)
    st.markdown("## Packages and services")

    pkg_df = pd.DataFrame([
        ["folium", "Interactive Leaflet maps", "HTML / browser"],
        ["streamlit-folium", "Render Folium inside Streamlit", "Streamlit app"],
        ["plotly", "Animated and analytical web maps", "PNG / HTML / PDF"],
        ["contextily", "Tile basemaps in matplotlib", "Static publication maps"],
        ["geopandas", "Vector GIS and CRS handling", "Spatial analysis"],
        ["earthengine-api", "Real Sentinel-2 RGB access", "Dynamic dated overlay"],
    ], columns=["Package", "Purpose", "Output"])
    st.dataframe(pkg_df, use_container_width=True, hide_index=True)

    tile_df = pd.DataFrame([
        ["Esri World Imagery", "Satellite basemap"],
        ["CartoDB Positron", "Clean neutral basemap"],
        ["CartoDB Dark", "Dark presentation basemap"],
        ["EOX Sentinel-2 Cloudless", "Reference tile overlay"],
        ["NASA GIBS MODIS", "Reference daily MODIS tiles"],
        ["EEA CORINE", "Reference CORINE tile service"],
    ], columns=["Service", "Role"])
    st.dataframe(tile_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
footer_label = (
    "real MODIS+GEDI+CORINE project CSVs"
    if data_src.startswith("📂")
    else "real DWD weather stations"
    if data_src.startswith("🌍")
    else "synthetic demo data"
)

st.markdown("---")
st.markdown(
    f"""
<div style="text-align:center; color:#6b7280; font-size:0.78rem; padding:10px 0 18px;">
    🛰️ Satellite Basemaps for Science Storytelling · Leipzig University /LIM/ ScaDS.AI · Data: {footer_label}
</div>
""",
    unsafe_allow_html=True,
)