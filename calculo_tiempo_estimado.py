from __future__ import annotations

import argparse
import json
import math
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import xarray as xr


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

MARINE_DIRS = {
    "oleaje": RAW_DIR / "marine_oleaje",
    "viento": RAW_DIR / "marine_viento",
    "corriente": RAW_DIR / "marine_corriente",
}

MARINE_WIND_DIR: Path = MARINE_DIRS["viento"]

WAVE_CANDIDATES = ["VHM0", "significant_wave_height", "swh", "hs"]

# Viento: alineado con NetCDF de data/raw/marine_viento 
# cmems_obs-wind_glo_phy_nrt_l3-
MPS_TO_KNOTS = 1.94384  # m/s → nudos (mismo factor que en el modelo náutico habitual)

# Escalares: prioridad explícita en nudos si existe; luego m/s (scatterometer, modelo, etc.).
WIND_SPEED_SCALAR_CANDIDATES = [
    "wind_speed_kts",
    "wspd_kts",
    "wind_speed",
    "se_model_speed",
    "wind_speed_10m",
    "si10",
    "ws10",
    "wspd",
    "wspd10",
    "sfcwind",
    "10si",
    "ff10",
    "wind",
]
# Dirección del viento en el .nc (nombre real de variable; p. ej. CMEMS usa wind_to_dir).
WIND_DIR_CANDIDATES = [
    "wind_dir",
    "wind_to_dir",
    "wind_direction",
    "model_wind_to_dir",
    "wind_from_direction",
    "wdir",
]
# Pares (u,v) coherentes; 
WIND_VECTOR_PAIRS: list[tuple[str, str]] = [
    ("eastward_wind", "northward_wind"),
    ("se_eastward_model_wind", "se_northward_model_wind"),
    ("u10", "v10"),
    ("10u", "10v"),
    ("u_10m", "v_10m"),
    ("u10m", "v10m"),
    ("u_wind", "v_wind"),
    ("eastward_wind_at_10m", "northward_wind_at_10m"),
]
CURRENT_U_CANDIDATES = ["uo", "u", "eastward_current"]
CURRENT_V_CANDIDATES = ["vo", "v", "northward_current"]


def find_latest_nc(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    files = list(directory.glob("*.nc"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def find_var(ds: xr.Dataset, candidates: list[str]) -> str | None:
    lower_map = {name.lower(): name for name in ds.data_vars}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def find_vector_vars(ds: xr.Dataset, u_candidates: list[str], v_candidates: list[str]) -> tuple[str | None, str | None]:
    return find_var(ds, u_candidates), find_var(ds, v_candidates)


TIME_DIM_NAMES = ("time", "t", "nominal_time", "forecast_time", "measurement_time")


def select_latest_time(da: xr.DataArray) -> xr.DataArray:
    for dim in TIME_DIM_NAMES:
        if dim in da.dims and da.sizes.get(dim, 0) > 1:
            return da.isel({dim: -1})
    return da


def get_lat_lon_names(ds: xr.Dataset) -> tuple[str | None, str | None]:
    lat_candidates = ["lat", "latitude", "nav_lat", "y"]
    lon_candidates = ["lon", "longitude", "nav_lon", "x"]
    lat_name = next((c for c in lat_candidates if c in ds.coords), None)
    lon_name = next((c for c in lon_candidates if c in ds.coords), None)
    return lat_name, lon_name


def sample_point(ds: xr.Dataset, lat: float, lon: float) -> xr.Dataset:
    lat_name, lon_name = get_lat_lon_names(ds)
    if not lat_name or not lon_name:
        raise ValueError("Dataset sin coordenadas lat/lon estándar.")
    return ds.sel({lat_name: lat, lon_name: lon}, method="nearest")


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    km = r_km * c
    return km / 1.852


def bearing_rad(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    bearing = math.atan2(y, x)
    return (bearing + 2 * math.pi) % (2 * math.pi)


def bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    br = bearing_rad(lat1, lon1, lat2, lon2)
    deg = math.degrees(br)
    return (deg + 360.0) % 360.0


def to_float(value: Any) -> float | None:
    try:
        arr = np.asarray(value, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        return float(np.nanmean(arr))
    except Exception:
        return None


# Dimensiones verticales habituales en productos CMEMS (uo/vo con varios niveles).
_DEPTH_DIM_NAMES = ("depth", "deptht", "lev", "level", "z", "depthu", "depthv", "sigma")


def _collapse_depth_to_surface(da: xr.DataArray) -> xr.DataArray:
    """Toma el primer nivel si queda dimensión vertical (p. ej. corriente en profundidad)."""
    out = da
    for dim in _DEPTH_DIM_NAMES:
        if dim in out.dims and out.sizes.get(dim, 0) > 1:
            out = out.isel({dim: 0})
        elif dim in out.dims:
            out = out.squeeze({dim}, drop=True)
    return out


def dataarray_to_scalar(da: xr.DataArray) -> float | None:
    """Un valor finito a partir de un punto muestreado (enmascarado, 0-D o pequeño array)."""
    arr = np.asarray(da.values, dtype=float)
    if not np.isfinite(arr).any():
        return None
    flat = arr[np.isfinite(arr)]
    return float(np.nanmean(flat))


def _wind_units_are_knots(units_raw: str) -> bool:
    s = (units_raw or "").strip().lower()
    if not s:
        return False
    if "knot" in s:
        return True
    if "m/s" in s or "m s-1" in s or "m s**-1" in s:
        return False
    return s in ("kt", "kn", "kts")


def _scalar_wind_mps_kts_from_nc(da: xr.DataArray, val: float, var_name: str) -> tuple[float, float]:
    """
    (m/s, nudos) valor muestreado del .nc y units / nombre de variable.
    """
    units = da.attrs.get("units")
    if isinstance(units, str) and _wind_units_are_knots(units):
        kts = val
        mps = kts / MPS_TO_KNOTS
        return mps, kts
    if isinstance(units, str) and not _wind_units_are_knots(units) and units.strip():
        mps = val
        return mps, mps * MPS_TO_KNOTS
    # Sin units claro: nombres típicos en nudos
    vn = var_name.lower()
    if "kts" in vn or "knot" in vn or vn.endswith("_kt"):
        kts = val
        return kts / MPS_TO_KNOTS, kts
    mps = val
    return mps, mps * MPS_TO_KNOTS


def _wind_speed_from_marine_viento_nc(
    lat1: float, lon1: float,
) -> tuple[float | None, float | None, str | None, str | None, str | None]:
    """
    Lee viento en el punto del barco (lat1, lon1) desde el último .nc en marine_viento.
    Devuelve (mps, kts, variable(s) usadas para velocidad
    """
    wind_path = find_latest_nc(MARINE_WIND_DIR)
    if not wind_path:
        return None, None, None, None, None

    fname = wind_path.name
    ds = xr.open_dataset(wind_path)
    try:
        wind_dir_var = find_var(ds, WIND_DIR_CANDIDATES)
        spd_name = find_var(ds, WIND_SPEED_SCALAR_CANDIDATES)
        if spd_name:
            da = select_latest_time(sample_point(ds, lat1, lon1)[spd_name])
            val = dataarray_to_scalar(da)
            if val is not None and val >= 0:
                mps, kts = _scalar_wind_mps_kts_from_nc(da, val, spd_name)
                return mps, kts, spd_name, fname, wind_dir_var

        for u_cand, v_cand in WIND_VECTOR_PAIRS:
            u_name = find_var(ds, [u_cand])
            v_name = find_var(ds, [v_cand])
            if not u_name or not v_name:
                continue
            u = select_latest_time(sample_point(ds, lat1, lon1)[u_name])
            v = select_latest_time(sample_point(ds, lat1, lon1)[v_name])
            u_val = dataarray_to_scalar(u)
            v_val = dataarray_to_scalar(v)
            if u_val is not None and v_val is not None:
                mps = math.sqrt(u_val**2 + v_val**2)
                kts = mps * MPS_TO_KNOTS
                return mps, kts, f"{u_name}+{v_name}", fname, wind_dir_var
    finally:
        ds.close()

    return None, None, None, fname, wind_dir_var


def estimate_conditions(lat1: float, lon1: float, lat2: float, lon2: float) -> dict[str, Any]:
    conditions: dict[str, Any] = {
        "wave_height_m": None,
        "wind_speed_mps": None,
        "wind_speed_kts": None,
        "current_along_mps": None,
        "wind_nc_variable": None,
        "wind_speed_nc_variable": None,
        "wind_nc_file": None,
        "wave_nc_file": None,
        "current_nc_file": None,
    }

    wave_path = find_latest_nc(MARINE_DIRS["oleaje"])
    conditions["wave_nc_file"] = wave_path.name if wave_path else None
    if wave_path:
        ds = xr.open_dataset(wave_path)
        try:
            var = find_var(ds, WAVE_CANDIDATES) or list(ds.data_vars)[0]
            da = _collapse_depth_to_surface(select_latest_time(sample_point(ds, lat1, lon1)[var]))
            val = dataarray_to_scalar(da)
            conditions["wave_height_m"] = val
        finally:
            ds.close()

    wspd, wkts, wvar_speed, wfile, wind_dir_var = _wind_speed_from_marine_viento_nc(lat1, lon1)
    conditions["wind_speed_mps"] = wspd
    conditions["wind_speed_kts"] = wkts
    conditions["wind_nc_variable"] = wind_dir_var
    conditions["wind_speed_nc_variable"] = wvar_speed
    conditions["wind_nc_file"] = wfile

    current_path = find_latest_nc(MARINE_DIRS["corriente"])
    conditions["current_nc_file"] = current_path.name if current_path else None
    if current_path:
        ds = xr.open_dataset(current_path)
        try:
            u_name, v_name = find_vector_vars(ds, CURRENT_U_CANDIDATES, CURRENT_V_CANDIDATES)
            if u_name and v_name:
                u = _collapse_depth_to_surface(select_latest_time(sample_point(ds, lat1, lon1)[u_name]))
                v = _collapse_depth_to_surface(select_latest_time(sample_point(ds, lat1, lon1)[v_name]))
                u_val = dataarray_to_scalar(u)
                v_val = dataarray_to_scalar(v)
                if u_val is None or v_val is None:
                    conditions["current_along_mps"] = None
                else:
                    brg = bearing_rad(lat1, lon1, lat2, lon2)
                    east_unit = math.sin(brg)
                    north_unit = math.cos(brg)
                    along = u_val * east_unit + v_val * north_unit
                    conditions["current_along_mps"] = along
        finally:
            ds.close()

    return conditions


def estimate_travel_time(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    boat_speed_kts: float | None = None,
) -> dict[str, Any]:
    distance_nm = haversine_nm(lat1, lon1, lat2, lon2)
    base_speed = float(os.getenv("MARINE_BASE_SPEED_KTS", "12"))
    wave_penalty = float(os.getenv("MARINE_WAVE_PENALTY_KTS_PER_M", "1.0"))
    wind_penalty = float(os.getenv("MARINE_WIND_PENALTY_KTS_PER_MPS", "0.5"))
    min_speed = float(os.getenv("MARINE_MIN_SPEED_KTS", "3"))

    # Velocidad de referencia: la medida actual del barco sustituye al valor por defecto 
    effective_base = float(boat_speed_kts) if boat_speed_kts is not None else base_speed

    conditions = estimate_conditions(lat1, lon1, lat2, lon2)
    wave = conditions["wave_height_m"] or 0.0
    wind = conditions["wind_speed_mps"] or 0.0
    current = conditions["current_along_mps"] or 0.0

    current_kts = current * 1.94384
    wave_loss_kts = wave * wave_penalty
    wind_loss_kts = wind * wind_penalty
    speed_uncapped = effective_base + current_kts - wave_loss_kts - wind_loss_kts
    speed_kts = max(speed_uncapped, min_speed)
    min_speed_floor_applied = speed_uncapped < min_speed

    hours = distance_nm / speed_kts if speed_kts > 0 else None
    minutes = hours * 60.0 if hours is not None else None
    brg_deg = bearing_degrees(lat1, lon1, lat2, lon2)
    distance_km = distance_nm * 1.852

    return {
        "distance_nm": round(distance_nm, 3),
        "distance_km": round(distance_km, 3),
        "bearing_degrees": round(brg_deg, 2),
        "estimated_hours": round(hours, 3) if hours is not None else None,
        "estimated_minutes": round(minutes, 2) if minutes is not None else None,
        "speed_knots": round(speed_kts, 3),
        "coordinates": {
            "start": {"lat": round(lat1, 6), "lon": round(lon1, 6)},
            "end": {"lat": round(lat2, 6), "lon": round(lon2, 6)},
        },
        "marine_data_available": {
            "wave": conditions["wave_height_m"] is not None,
            "wind": conditions["wind_speed_mps"] is not None,
            "current": conditions["current_along_mps"] is not None,
        },
        "nc_files_used": {
            "marine_oleaje": conditions.get("wave_nc_file"),
            "marine_viento": conditions.get("wind_nc_file"),
            "marine_corriente": conditions.get("current_nc_file"),
        },
        "adjustments_knots": {
            "effective_base": round(effective_base, 3),
            "current_along_track": round(current_kts, 3),
            "wave_penalty": round(wave_loss_kts, 3),
            "wind_penalty": round(wind_loss_kts, 3),
            "before_min_floor": round(speed_uncapped, 3),
        },
        "limits": {
            "min_speed_knots": round(min_speed, 3),
            "min_speed_floor_applied": min_speed_floor_applied,
        },
        "model_parameters": {
            "base_speed_knots_default": round(base_speed, 3),
            "wave_penalty_kts_per_m": round(wave_penalty, 4),
            "wind_penalty_kts_per_mps": round(wind_penalty, 4),
        },
        "inputs": {
            "base_speed_knots": base_speed,
            "boat_speed_knots": None if boat_speed_kts is None else round(float(boat_speed_kts), 3),
            "effective_base_knots": round(effective_base, 3),
            "wave_height_m": None if conditions["wave_height_m"] is None else round(wave, 3),
            "wind_speed_mps": None if conditions["wind_speed_mps"] is None else round(wind, 3),
            "wind_speed_kts": None
            if conditions.get("wind_speed_kts") is None
            else round(float(conditions["wind_speed_kts"]), 3),
            "wave_nc_file": conditions.get("wave_nc_file"),
            "wind_nc_file": conditions.get("wind_nc_file"),
            "current_nc_file": conditions.get("current_nc_file"),
            "wind_nc_variable": conditions.get("wind_nc_variable"),
            "wind_speed_nc_variable": conditions.get("wind_speed_nc_variable"),
            "current_along_mps": None if conditions["current_along_mps"] is None else round(current, 3),
            "current_along_kts": None if conditions["current_along_mps"] is None else round(current_kts, 3),
        },
    }


class EstimateHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  
        parsed = urlparse(self.path)
        if parsed.path not in ("/", "/estimate"):
            self.send_error(404, "Not Found")
            return

        params = parse_qs(parsed.query)
        try:
            lat1 = float(params.get("start_lat", [None])[0])
            lon1 = float(params.get("start_lon", [None])[0])
            lat2 = float(params.get("end_lat", [None])[0])
            lon2 = float(params.get("end_lon", [None])[0])
        except Exception:
            self.send_error(400, "Missing or invalid coordinates.")
            return

        boat_speed_kts: float | None = None
        for key in ("boat_speed_kts", "speed_kts", "boat_speed"):
            raw = params.get(key, [None])[0]
            if raw is not None and str(raw).strip() != "":
                boat_speed_kts = to_float(raw)
                if boat_speed_kts is None or boat_speed_kts < 0:
                    self.send_error(400, "Invalid boat speed (non-negative number expected).")
                    return
                break

        try:
            result = estimate_travel_time(lat1, lon1, lat2, lon2, boat_speed_kts=boat_speed_kts)
        except Exception as exc: 
            self.send_error(500, str(exc))
            return

        payload = json.dumps(result, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def run_server(host: str, port: int) -> None:
    server = HTTPServer((host, port), EstimateHandler)
    print(f"API activa en http://{host}:{port}/estimate")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="API de estimación de tiempo de navegación.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--start-lat", type=float)
    parser.add_argument("--start-lon", type=float)
    parser.add_argument("--end-lat", type=float)
    parser.add_argument("--end-lon", type=float)
    parser.add_argument(
        "--boat-speed",
        type=float,
        default=None,
        metavar="KTS",
        help="Velocidad actual del barco en nudos (sustituye a MARINE_BASE_SPEED_KTS en el modelo).",
    )
    args = parser.parse_args()

    if None not in (args.start_lat, args.start_lon, args.end_lat, args.end_lon):
        if args.boat_speed is not None and args.boat_speed < 0:
            parser.error("--boat-speed must be non-negative.")
        result = estimate_travel_time(
            args.start_lat,
            args.start_lon,
            args.end_lat,
            args.end_lon,
            boat_speed_kts=args.boat_speed,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
