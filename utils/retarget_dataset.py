import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from shapely.geometry import Point, shape


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _load_embedded_json(value: Any, field_name: str) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"{field_name} is missing or not a JSON object")


def _prepare_hexagons(hexagon_geojson: Dict[str, Any]) -> List[Tuple[int, Any]]:
    features = hexagon_geojson.get("features")
    if not isinstance(features, list):
        raise ValueError("hexagonGeoJson.features must be an array")

    polygons: List[Tuple[int, Any]] = []
    for idx, feature in enumerate(features):
        geometry = feature.get("geometry") if isinstance(feature, dict) else None
        if not geometry:
            continue
        polygons.append((idx, shape(geometry)))
    return polygons


def _find_hexagon_index(point_lon: float, point_lat: float, polygons: List[Tuple[int, Any]]) -> int:
    point = Point(point_lon, point_lat)
    for idx, polygon in polygons:
        if polygon.contains(point) or polygon.touches(point):
            return idx
    return -1


def _extract_sensor_attachments(network_geojson: Dict[str, Any]) -> List[Dict[str, Any]]:
    features = network_geojson.get("features")
    if not isinstance(features, list):
        raise ValueError("networkGeoJson.features must be an array")

    attachments: List[Dict[str, Any]] = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        properties = feature.get("properties")
        if not isinstance(properties, dict):
            continue
        if properties.get("type") != "Node":
            continue

        node_id = str(properties.get("id", "")).strip()
        linked_sensor = properties.get("linked_sensor")
        if (
            properties.get("measurementTag")
            and node_id
            and linked_sensor is not None
            and str(linked_sensor).strip() != ""
        ):
            try:
                sensor_id = int(linked_sensor)
            except (TypeError, ValueError):
                continue
            attachments.append({"nodeId": node_id, "sensorId": sensor_id})

    return attachments


def _relabel_sensor_ids(attachments: List[Dict[str, Any]], sensors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sensor_id_lookup = {str(item["nodeId"]): item["sensorId"] for item in attachments}

    updated: List[Dict[str, Any]] = []
    for sensor in sensors:
        if not isinstance(sensor, dict):
            continue
        node_id = str(sensor.get("nodeId", "")).strip()
        next_sensor = dict(sensor)
        next_sensor["sensorId"] = sensor_id_lookup.get(node_id)
        updated.append(next_sensor)
    return updated


def retarget_dataset(
    dataset_payload: Dict[str, Any],
    pipeline_payload: Dict[str, Any],
    *,
    relabel_sensor_ids: bool,
) -> Dict[str, Any]:
    dataset = dataset_payload.get("dataset")
    if not isinstance(dataset, list):
        raise ValueError("dataset payload must contain a dataset array")

    hexagon_geojson = _load_embedded_json(pipeline_payload.get("hexagonGeoJson"), "hexagonGeoJson")
    network_geojson = _load_embedded_json(pipeline_payload.get("networkGeoJson"), "networkGeoJson")

    polygons = _prepare_hexagons(hexagon_geojson)
    attachments = _extract_sensor_attachments(network_geojson)

    updated_dataset: List[Dict[str, Any]] = []
    touched_hex_runs = 0
    touched_sensor_runs = 0

    for run in dataset:
        if not isinstance(run, dict):
            continue
        updated_run = dict(run)

        leak_location = run.get("leakLocationWgs84")
        new_hexagon_id = -1
        if isinstance(leak_location, dict):
            longitude = leak_location.get("longitude")
            latitude = leak_location.get("latitude")
            if longitude is not None and latitude is not None:
                new_hexagon_id = _find_hexagon_index(float(longitude), float(latitude), polygons)
        if updated_run.get("hexagonId") != new_hexagon_id:
            touched_hex_runs += 1
        updated_run["hexagonId"] = new_hexagon_id

        if relabel_sensor_ids:
            sensors = run.get("sensors")
            if isinstance(sensors, list):
                relabeled = _relabel_sensor_ids(attachments, sensors)
                if relabeled != sensors:
                    touched_sensor_runs += 1
                updated_run["sensors"] = relabeled

        updated_dataset.append(updated_run)

    updated_payload = dict(dataset_payload)
    updated_payload["dataset"] = updated_dataset
    updated_payload["allHexagonIds"] = sorted(
        {
            int(run.get("hexagonId"))
            for run in updated_dataset
            if isinstance(run, dict)
            and run.get("hexagonId") is not None
            and int(run.get("hexagonId")) >= 0
        }
    )
    updated_payload["retargeting"] = {
        "hexagonRunsUpdated": touched_hex_runs,
        "sensorRunsUpdated": touched_sensor_runs,
        "attachmentCount": len(attachments),
        "hexagonCount": len(polygons),
    }
    return updated_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retarget an existing leakage dataset to a new saved geojson payload from the same INP."
    )
    parser.add_argument("--dataset", required=True, help="Path to existing train_data.json or merged dataset JSON")
    parser.add_argument("--geojson", required=True, help="Path to the new saved pipeline payload JSON")
    parser.add_argument("--out", required=True, help="Path for rewritten dataset JSON")
    parser.add_argument(
        "--keep-sensor-ids",
        action="store_true",
        help="Do not rewrite sensorId annotations from the new payload's linked sensors",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    payload_path = Path(args.geojson)
    out_path = Path(args.out)

    dataset_payload = _load_json(dataset_path)
    pipeline_payload = _load_json(payload_path)
    updated_payload = retarget_dataset(
        dataset_payload,
        pipeline_payload,
        relabel_sensor_ids=not args.keep_sensor_ids,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(updated_payload, handle, indent=2)
        handle.write("\n")

    meta = updated_payload["retargeting"]
    print(
        f"wrote {out_path}: hexagon runs updated={meta['hexagonRunsUpdated']}, "
        f"sensor runs updated={meta['sensorRunsUpdated']}, "
        f"hexagons={meta['hexagonCount']}, attachments={meta['attachmentCount']}"
    )


if __name__ == "__main__":
    main()