import json
from typing import Any, Dict, List, Tuple
import logging
from shapely.geometry import Point, shape

class LeakageGenerationError(Exception):
    pass

def _load_embedded_json(value: Any, field_name: str) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"{field_name} is missing or not a JSON object")

def _prepare_hexagons(hexagon_geojson: Dict):
    if not isinstance(hexagon_geojson, dict):
        raise LeakageGenerationError("hexagonGeoJson must be an object")

    features = hexagon_geojson.get("features")
    if not isinstance(features, list):
        raise LeakageGenerationError("hexagonGeoJson.features must be an array")

    polygons = []
    for idx, feature in enumerate(features):
        geometry = feature.get("geometry") if isinstance(feature, dict) else None
        if not geometry:
            continue
        try:
            polygons.append((idx, shape(geometry)))
        except Exception:
            continue
    return polygons


#same helper as generator
def _find_hexagon_index(point_lon: float, point_lat: float, polygons) -> int:
    point = Point(point_lon, point_lat)
    for idx, poly in polygons:
        if poly.contains(point) or poly.touches(point):
            return idx
    return -1


def _relabel_run_hexagon(run: Dict[str, Any], polygons: List[Tuple[int, Any]]) -> int:
    leak_location = run.get("leakLocationWgs84")
    if not isinstance(leak_location, dict):
        return -1

    longitude = leak_location.get("longitude")
    latitude = leak_location.get("latitude")
    if longitude is None or latitude is None:
        return -1

    return _find_hexagon_index(float(longitude), float(latitude), polygons)


def retarget_dataset(
    dataset_payload: Dict[str, Any],
    pipeline_payload: Dict[str, Any],
) -> Dict[str, Any]:
    logging.info("retargeting the dataset")
    #payload has summary, dataset and failure, get dataset and validate it
    dataset = dataset_payload.get("dataset")
    if not isinstance(dataset, list):
        raise ValueError("dataset payload must contain a dataset array")

    #get hexagonjson from the pipeline_payload (after the first three steps)
    hexagon_geojson = _load_embedded_json(pipeline_payload.get("hexagonGeoJson"), "hexagonGeoJson")
    #same as generator's helper function
    polygons = _prepare_hexagons(hexagon_geojson)

    updated_dataset: List[Dict[str, Any]] = []
    touched_hex_runs = 0

    for run in dataset:
        if not isinstance(run, dict):
            continue
        updated_run = dict(run)

        new_hexagon_id = _relabel_run_hexagon(run, polygons)
        if updated_run.get("hexagonId") != new_hexagon_id:
            touched_hex_runs += 1
        updated_run["hexagonId"] = new_hexagon_id

        updated_dataset.append(updated_run)

    updated_payload = dict(dataset_payload)
    updated_payload["hexagonGeoJson"] = pipeline_payload.get("hexagonGeoJson")
    updated_payload["dataset"] = updated_dataset

    logging.info(f"retargetting done, {touched_hex_runs} runs changed")


    return updated_payload
