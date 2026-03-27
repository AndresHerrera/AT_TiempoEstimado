from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import rasterio
from copernicusmarine import (  # type: ignore
    CoordinatesOutOfDatasetBounds,
    VariableDoesNotExistInTheDataset,
    subset,
)
from dotenv import dotenv_values, load_dotenv
from rasterio.transform import from_bounds
import xarray as xr


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
AOI_FILE = BASE_DIR / "area_influencia.geojson"
ENV_FILE = BASE_DIR / ".env"


def resolve_aoi_geojson_path() -> Path:
    """Ruta al GeoJSON del área de influencia (override con MARINE_AOI_GEOJSON)."""
    raw = os.getenv("MARINE_AOI_GEOJSON", "").strip()
    return Path(raw) if raw else AOI_FILE

THEMES = [
    {
        "label": "marine_oleaje",
        "dataset_env": "MARINE_WAVE_DATASET_ID",
        "variables_env": "MARINE_WAVE_VARIABLES",
        "default_dataset_id": "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i",
        "default_variables": ["VHM0", "VHM0_SW1", "VHM0_SW2", "significant_wave_height"],
    },
    {
        # Global Ocean Daily Gridded Sea Surface Winds from Scatterometer (KNMI / CMEMS).
        # Product ID (catálogo): WIND_GLO_PHY_L3_NRT_012_002 — la API subset() exige el dataset_id
        "label": "marine_viento",
        "product_name": "Global Ocean Daily Gridded Sea Surface Winds from Scatterometer",
        "product_id": "WIND_GLO_PHY_L3_NRT_012_002",
        "dataset_env": "MARINE_WIND_DATASET_ID",
        "variables_env": "MARINE_WIND_VARIABLES",
        "default_dataset_id": "cmems_obs-wind_glo_phy_nrt_l3-metopb-ascat-des-0.125deg_P1D-i",
        "default_variables": [
            "wind_speed",
            "eastward_wind",
            "northward_wind",
            "wind_to_dir",
        ],
    },
    {
        # Corrientes oceánicas en rejilla — medias diarias (Global Ocean Physics Analysis & Forecast).
        # Dataset: cmems_mod_glo_phy_anfc_0.083deg_P1D-m (P1D = daily).
        "label": "marine_corriente",
        "dataset_env": "MARINE_CURRENT_DATASET_ID",
        "variables_env": "MARINE_CURRENT_VARIABLES",
        "default_dataset_id": "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        "default_variables": [
            "uo",
            "vo",
            "zos",
            "thetao",
        ],
    },
]


def ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def load_env_credentials() -> tuple[str, str]:
    load_dotenv(dotenv_path=ENV_FILE, override=True)
    username = os.getenv("COPERNICUS_USERNAME")
    password = os.getenv("COPERNICUS_PASSWORD")
    if not username or not password:
        values = dotenv_values(ENV_FILE)
        username = username or values.get("COPERNICUS_USERNAME")
        password = password or values.get("COPERNICUS_PASSWORD")
    if not username or not password:
        raise RuntimeError(f"Faltan credenciales en {ENV_FILE}.")
    return username, password


def load_aoi_bounds(geojson_path: Path | None = None) -> tuple[float, float, float, float]:
    """
    Bounding box WGS84 (min_lon, min_lat, max_lon, max_lat) del AOI.
    """
    path = geojson_path or resolve_aoi_geojson_path()
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo de área de influencia: {path}")
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"El GeoJSON no contiene geometrías: {path}")
    geom = gdf.to_crs("EPSG:4326").union_all()
    minx, miny, maxx, maxy = geom.bounds
    return minx, miny, maxx, maxy


def parse_variables(raw: str | None, defaults: Iterable[str]) -> list[str] | None:
    if raw:
        items = [item.strip() for item in raw.split(",") if item.strip()]
        return items or None
    return list(defaults) if defaults else None


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_iso_datetime(token: str) -> datetime:
    t = token.strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    return datetime.fromisoformat(t)


def parse_dataset_time_bounds_from_subset_error(message: str) -> tuple[datetime, datetime] | None:
    """
    Extrae [t_min, t_max] del mensaje CoordinatesOutOfDatasetBounds del toolbox
    """
    m = re.search(r"dataset coordinates \[([^\]]+)\]", message, re.IGNORECASE)
    if not m:
        return None
    inner = m.group(1)
    parts = [p.strip() for p in inner.split(",")]
    if len(parts) != 2:
        return None
    try:
        a = _parse_iso_datetime(parts[0])
        b = _parse_iso_datetime(parts[1])
    except ValueError:
        return None
    lo, hi = (a, b) if a <= b else (b, a)
    return _ensure_utc(lo), _ensure_utc(hi)


def clip_time_range_to_dataset(
    start_dt: datetime,
    end_dt: datetime,
    ds_lo: datetime,
    ds_hi: datetime,
) -> tuple[datetime, datetime]:
    """Intersección de [start_dt, end_dt] con la ventana temporal del dataset."""
    s = _ensure_utc(start_dt)
    e = _ensure_utc(end_dt)
    lo = _ensure_utc(ds_lo)
    hi = _ensure_utc(ds_hi)
    ns = max(s, lo)
    ne = min(e, hi)
    if ns < ne:
        return ns, ne
    return lo, hi


def choose_variable(ds: xr.Dataset, candidates: Iterable[str] | None) -> str | None:
    if candidates:
        lower = {name.lower(): name for name in ds.data_vars}
        for cand in candidates:
            key = cand.lower()
            if key in lower:
                return lower[key]
    data_vars = list(ds.data_vars)
    return data_vars[0] if data_vars else None


def export_dataset_to_tif(
    ds: xr.Dataset,
    var_name: str,
    out_path: Path,
    nodata: float | int | None = None,
) -> None:
    da = ds[var_name]

    lat_name = None
    lon_name = None
    for cand in ("lat", "latitude", "y"):
        if cand in ds.coords:
            lat_name = cand
            break
    for cand in ("lon", "longitude", "x"):
        if cand in ds.coords:
            lon_name = cand
            break
    if lat_name is None or lon_name is None:
        raise ValueError("No se encontraron coordenadas lat/lon estándar en el dataset.")

    # Seleccionar primeras posiciones para dimensiones no espaciales
    spatial_dims = {lat_name, lon_name}
    for dim in list(da.dims):
        if dim not in spatial_dims:
            da = da.isel({dim: 0})

    data = da.values
    if data.ndim != 2:
        raise ValueError(f"La variable {var_name} no tiene forma 2D tras recortar.")

    lat_vals = np.asarray(ds[lat_name].values)
    lon_vals = np.asarray(ds[lon_name].values)

    if lat_vals.ndim == 1 and lon_vals.ndim == 1:
        miny, maxy = float(lat_vals.min()), float(lat_vals.max())
        minx, maxx = float(lon_vals.min()), float(lon_vals.max())
    else:
        miny, maxy = float(lat_vals.min()), float(lat_vals.max())
        minx, maxx = float(lon_vals.min()), float(lon_vals.max())

    transform = from_bounds(minx, miny, maxx, maxy, data.shape[1], data.shape[0])
    if nodata is not None:
        data = np.where(np.isnan(data), nodata, data)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def download_theme(
    label: str,
    dataset_id: str,
    variables: list[str] | None,
    bounds: tuple[float, float, float, float],
    start_dt: datetime,
    end_dt: datetime,
    username: str,
    password: str,
) -> None:
    minx, miny, maxx, maxy = bounds
    out_dir = RAW_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    cur_start, cur_end = start_dt, end_dt

    def out_nc_path() -> Path:
        return out_dir / f"{label}_{cur_start:%Y%m%dT%H%M%S}_{cur_end:%Y%m%dT%H%M%S}.nc"

    def do_subset(t0: datetime, t1: datetime, vars_: list[str] | None) -> None:
        subset(
            dataset_id=dataset_id,
            username=username,
            password=password,
            variables=vars_,
            minimum_longitude=minx,
            maximum_longitude=maxx,
            minimum_latitude=miny,
            maximum_latitude=maxy,
            start_datetime=t0,
            end_datetime=t1,
            output_filename=out_nc_path().name,
            output_directory=out_dir,
            file_format="netcdf",
            overwrite=True,
        )

    def run_subset_with_time_clip(vars_: list[str] | None) -> None:
        nonlocal cur_start, cur_end
        try:
            do_subset(cur_start, cur_end, vars_)
        except CoordinatesOutOfDatasetBounds as exc:
            msg = str(exc)
            if "time" not in msg.lower():
                raise
            bounds_t = parse_dataset_time_bounds_from_subset_error(msg)
            if bounds_t is None:
                raise
            ds_lo, ds_hi = bounds_t
            cur_start, cur_end = clip_time_range_to_dataset(cur_start, cur_end, ds_lo, ds_hi)
            print(
                f"  Rango temporal ajustado al disponible en el dataset: "
                f"{cur_start.isoformat()} — {cur_end.isoformat()}"
            )
            do_subset(cur_start, cur_end, vars_)

    print(f"Descargando {label} desde {dataset_id} ...")
    try:
        run_subset_with_time_clip(variables)
    except VariableDoesNotExistInTheDataset:
        print(
            f"  Alguna variable pedida no existe en este dataset ({label}). "
            "Reintentando sin filtro de variables..."
        )
        run_subset_with_time_clip(None)
    except Exception as exc: 
        msg = str(exc)
        if "VariableDoesNotExistInTheDataset" in type(exc).__name__ or (
            "neither a variable" in msg.lower() and "standard name" in msg.lower()
        ):
            print(
                f"  Variables no disponibles para {label}. "
                "Reintentando sin filtro de variables..."
            )
            run_subset_with_time_clip(None)
        else:
            raise

    print(f"  Descarga completada en {out_nc_path()}")


def main() -> None:
    ensure_dirs()
    username, password = load_env_credentials()
    aoi_path = resolve_aoi_geojson_path()
    bounds = load_aoi_bounds(aoi_path)
    minx, miny, maxx, maxy = bounds
    print(
        f"Área de influencia: {aoi_path.resolve()} (EPSG:4326 bbox → "
        f"lon [{minx:.6f}, {maxx:.6f}], lat [{miny:.6f}, {maxy:.6f}])"
    )
    print("Todas las descargas (oleaje, viento, corriente) usan este mismo recorte.")
    # Por defecto descarga lo más reciente (último día)
    days_back = int(os.getenv("MARINE_DAYS_BACK", "1"))
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days_back)

    for theme in THEMES:
        dataset_id = os.getenv(theme["dataset_env"]) or theme["default_dataset_id"]
        variables = parse_variables(os.getenv(theme["variables_env"]), theme["default_variables"])
        download_theme(
            label=theme["label"],
            dataset_id=dataset_id,
            variables=variables,
            bounds=bounds,
            start_dt=start_dt,
            end_dt=end_dt,
            username=username,
            password=password,
        )


if __name__ == "__main__":
    main()
