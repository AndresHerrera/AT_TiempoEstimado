# AT_TiempoEstimado

Herramienta para estimar el tiempo de navegación entre dos coordenadas usando:

- distancia geodésica (haversine),
- datos marinos recientes de Copernicus Marine (oleaje, viento y corriente),
- una velocidad base del barco (o velocidad medida en tiempo real).

El proyecto incluye:

- `download_copernicusmarine_data.py`: descarga datos `.nc` recortados al área de influencia.
- `calculo_tiempo_estimado.py`: calcula la estimación por línea de comandos o expone una API HTTP simple.

## Requisitos

- Python 3.10
- Cuenta en Copernicus Marine con credenciales activas

Dependencias principales: `xarray`, `copernicusmarine`, `geopandas`, `rasterio`, `python-dotenv`, `netCDF4`.

## Instalación

```bash
conda create -n at_piloto python=3.10
conda activate at_piloto
pip install -r requirements.txt
```

## Configuración

### 1) Credenciales en `.env`

Crear (o completar) el archivo `.env` en la raíz del proyecto:

```env
COPERNICUS_USERNAME=tu_usuario
COPERNICUS_PASSWORD=tu_password
```

### 2) Área de influencia

Por defecto se usa `area_influencia.geojson`.  
Opcionalmente puedes cambiarlo con:

```env
MARINE_AOI_GEOJSON=ruta/al/archivo.geojson
```

## Flujo recomendado

### Paso 1: Descargar datos del día

```bash
python download_copernicusmarine_data.py
```

Se guardan en:

- `data/raw/marine_oleaje`
- `data/raw/marine_viento`
- `data/raw/marine_corriente`

### Paso 2A: Ejecutar cálculo directo por CLI

```bash
python calculo_tiempo_estimado.py --start-lat 40.59509 --start-lon 5.304011 --end-lat 41.30055 --end-lon 2.21143 --boat-speed 16
```

### Paso 2B: Levantar API local

```bash
python calculo_tiempo_estimado.py --host 127.0.0.1 --port 8000
```

Endpoint:

- `GET /estimate`

Parámetros requeridos:

- `start_lat`
- `start_lon`
- `end_lat`
- `end_lon`

Parámetros opcionales para velocidad del barco (nudos):

- `boat_speed_kts`
- `speed_kts`
- `boat_speed`

Ejemplo:

```text
http://127.0.0.1:8000/estimate?start_lat=40.59509&start_lon=5.304011&end_lat=41.30055&end_lon=2.21143&boat_speed=16
```

## Ejemplos adicionales

Coordenadas:

- inicio barco: `5.30401,40.59509`
- destino puerto: `2.21143,41.30055`

Caso alternativo:

- puerto destino: `0.11819,40.11625`
- barco inicio: `5.20282,39.82994`

Consulta API:

```text
http://127.0.0.1:8000/estimate?start_lat=39.82994&start_lon=5.20282&end_lat=40.11625&end_lon=0.11819&boat_speed=16
```

## Variables de entorno opcionales

Parámetros del modelo (si no se definen, se usan valores por defecto):

- `MARINE_BASE_SPEED_KTS` (default: `12`)
- `MARINE_WAVE_PENALTY_KTS_PER_M` (default: `1.0`)
- `MARINE_WIND_PENALTY_KTS_PER_MPS` (default: `0.5`)
- `MARINE_MIN_SPEED_KTS` (default: `3`)

Parámetros de descarga:

- `MARINE_DAYS_BACK` (default: `1`)
- `MARINE_WAVE_DATASET_ID`, `MARINE_WAVE_VARIABLES`
- `MARINE_WIND_DATASET_ID`, `MARINE_WIND_VARIABLES`
- `MARINE_CURRENT_DATASET_ID`, `MARINE_CURRENT_VARIABLES`

## Notas

- Si no hay datos recientes para alguna capa (oleaje, viento o corriente), la estimación se calcula con la información disponible.
- El servicio toma automáticamente el archivo `.nc` más reciente de cada carpeta.
