import os
import random
import tempfile
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import wntr
from pyproj import Transformer
from shapely.geometry import Point, shape
import logging
import time
from leakgen.utils import *

class LeakageGenerationError(Exception):
    pass


def _parse_demand_range(leak_demand_range: List[float]) -> List[float]:
    if not isinstance(leak_demand_range, list) or len(leak_demand_range) != 3:
        raise LeakageGenerationError("leakDemandRange must be [start, end, step]")

    try:
        start = float(leak_demand_range[0])
        end = float(leak_demand_range[1])
        step = float(leak_demand_range[2])
    except (TypeError, ValueError):
        raise LeakageGenerationError("leakDemandRange must contain numeric values")

    if step <= 0:
        raise LeakageGenerationError("leakDemandRange step must be greater than 0")
    if end < start:
        raise LeakageGenerationError("leakDemandRange end must be >= start")

    values = []
    current = start
    epsilon = step / 1_000_000
    while current <= end + epsilon:
        values.append(round(current, 6))
        current += step

    return values


def _normalize_attachments(step3_sensor_attachments: List[Dict]) -> List[Dict]:
    normalized = []
    for item in step3_sensor_attachments:
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("nodeId", "")).strip()
        sensor_id = item.get("sensorId")
        if not node_id or sensor_id is None:
            continue
        normalized.append({"nodeId": node_id, "sensorId": sensor_id})
    return normalized


def _load_wn_from_inp(network_inp_content: str):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".inp", delete=False) as tmp:
            tmp.write(network_inp_content)
            temp_path = tmp.name
        return wntr.network.WaterNetworkModel(temp_path)
    except Exception as exc:
        raise LeakageGenerationError(f"Failed to parse INP content: {exc}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


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


def _build_wgs84_node_coordinates(wn, source_crs: str) -> Dict[str, Tuple[float, float]]:
    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    node_coords = {}
    for node_name in wn.node_name_list:
        node = wn.get_node(node_name)
        coords = getattr(node, "coordinates", None)
        if coords is None or len(coords) < 2:
            continue
        lon, lat = transformer.transform(float(coords[0]), float(coords[1]))
        node_coords[str(node_name)] = (lon, lat)
    return node_coords


def _build_pipe_paths_source_crs(wn) -> Dict[str, List[Tuple[float, float]]]:
    """Build per-pipe polylines in source CRS using start/end nodes and optional vertices."""
    paths: Dict[str, List[Tuple[float, float]]] = {}

    for pipe_name in wn.pipe_name_list:
        pipe = wn.get_link(pipe_name)

        start_node = wn.get_node(pipe.start_node_name)
        end_node = wn.get_node(pipe.end_node_name)
        start_coords = getattr(start_node, "coordinates", None)
        end_coords = getattr(end_node, "coordinates", None)
        if (
            start_coords is None
            or end_coords is None
            or len(start_coords) < 2
            or len(end_coords) < 2
        ):
            continue

        points: List[Tuple[float, float]] = [
            (float(start_coords[0]), float(start_coords[1]))
        ]

        vertices = getattr(pipe, "vertices", None)
        if vertices is None:
            vertices = []

        for vertex in vertices:
            if vertex is None or len(vertex) < 2:
                continue
            points.append((float(vertex[0]), float(vertex[1])))

        points.append((float(end_coords[0]), float(end_coords[1])))

        # Remove adjacent duplicates to avoid zero-length segments.
        deduped: List[Tuple[float, float]] = []
        for pt in points:
            if not deduped or deduped[-1] != pt:
                deduped.append(pt)

        if len(deduped) >= 2:
            paths[str(pipe_name)] = deduped

    return paths


def _point_along_polyline(points: List[Tuple[float, float]], fraction: float) -> Tuple[float, float]:
    if points is None or len(points) < 2:
        raise LeakageGenerationError("Pipe geometry has insufficient points")

    clamped_fraction = max(0.0, min(1.0, fraction))

    segments: List[Tuple[Tuple[float, float], Tuple[float, float], float]] = []
    total_length = 0.0
    for idx in range(len(points) - 1):
        p0 = points[idx]
        p1 = points[idx + 1]
        seg_len = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
        segments.append((p0, p1, seg_len))
        total_length += seg_len

    if total_length <= 0:
        return points[0]

    target = clamped_fraction * total_length
    traversed = 0.0

    for p0, p1, seg_len in segments:
        if seg_len <= 0:
            continue
        if traversed + seg_len >= target:
            local = (target - traversed) / seg_len
            x = p0[0] + (p1[0] - p0[0]) * local
            y = p0[1] + (p1[1] - p0[1]) * local
            return (x, y)
        traversed += seg_len

    return points[-1]


def _find_hexagon_index(point_lon: float, point_lat: float, polygons) -> int:
    point = Point(point_lon, point_lat)
    for idx, poly in polygons:
        if poly.contains(point) or poly.touches(point):
            return idx
    return -1


def _pipe_positions(pipe_length: float, step_distance: float) -> List[float]:
    if pipe_length <= 0 or step_distance <= 0:
        return []

    positions = []
    current = step_distance
    while current < pipe_length:
        positions.append(current)
        current += step_distance

    return positions


def _split_pipe_with_leak(wn, pipe_name: str, position: float, run_idx: int):
    pipe = wn.get_link(pipe_name)
    if pipe.length <= 0:
        raise LeakageGenerationError(f"Pipe {pipe_name} has non-positive length")

    fraction = max(0.0, min(1.0, position / pipe.length))

    leak_junction_name = f"LEAK_J_{run_idx}"
    new_pipe_name = f"LEAK_P_{run_idx}"

    split_result = None
    try:
        split_result = wntr.morph.split_pipe(
            wn,
            pipe_name_to_split=pipe_name,
            new_pipe_name=new_pipe_name,
            new_junction_name=leak_junction_name,
            add_pipe_at_end=True,
            split_at_point=fraction,
        )
    except TypeError:
        split_result = wntr.morph.split_pipe(
            wn,
            pipe_name,
            new_pipe_name,
            leak_junction_name,
            True,
            fraction,
        )

    # WNTR versions differ: some mutate in place, others return a new network object.
    wn_after_split = wn
    if split_result is not None:
        if hasattr(split_result, "node_name_list"):
            wn_after_split = split_result
        elif isinstance(split_result, tuple):
            returned_wn = next(
                (item for item in split_result if hasattr(item, "node_name_list")),
                None,
            )
            if returned_wn is not None:
                wn_after_split = returned_wn

    if leak_junction_name not in wn_after_split.node_name_list:
        raise LeakageGenerationError(
            f"Leak junction {leak_junction_name} was not created after splitting pipe {pipe_name}"
        )

    return wn_after_split, leak_junction_name


def _extract_sensor_pressures(results, attachments: List[Dict], noise_amplitude: float = 0.05):
    pressures_df = results.node["pressure"]
    if pressures_df.empty:
        raise LeakageGenerationError("EPANET produced no pressure results – simulation may have failed to converge")
    times = [float(t) for t in list(pressures_df.index)]

    # Build a lookup from nodeId -> sensorId for nodes that have a linked sensor
    sensor_id_lookup = {att["nodeId"]: att["sensorId"] for att in attachments}

    sensors = []
    for node_id in pressures_df.columns:
        raw_pressure_values = [float(v) for v in pressures_df[node_id].tolist()]
        timesteps = len(raw_pressure_values)
        if timesteps > 0:
            noise = noise_amplitude * np.random.randn(timesteps)
            pressure_values = [float(v + n) for v, n in zip(raw_pressure_values, noise)]
        else:
            pressure_values = []

        sensors.append(
            {
                "sensorId": sensor_id_lookup.get(str(node_id)),
                "nodeId": str(node_id),
                "pressure": pressure_values,
            }
        )

    return times, sensors


def _apply_noise_to_base_pressures(base_pressures: Dict[str, List[float]], times: List[float], attachments: List[Dict], noise_amplitude: float = 0.05):
    # Build a lookup from nodeId -> sensorId for nodes that have a linked sensor
    sensor_id_lookup = {att["nodeId"]: att["sensorId"] for att in attachments}

    sensors = []
    for node_id, raw in base_pressures.items():
        if len(raw) > 0:
            timesteps = len(raw)
            noise = noise_amplitude * np.random.randn(timesteps)
            pressure_values = [float(v + n) for v, n in zip(raw, noise)]
        else:
            pressure_values = []
        sensors.append(
            {
                "sensorId": sensor_id_lookup.get(node_id),
                "nodeId": node_id,
                "pressure": pressure_values,
            }
        )
    return times, sensors


def generate_leakage_dataset(
    *,
    hexagon_geojson: Dict,
    network_inp_content: str,
    crs: str,
    delta_x_leak: float,
    hexagon_radius: float,
    step3_sensor_attachments: List[Dict],
    leak_demand_range: List[float],
    no_leak_portion: float = 0.0,
    noise_amplitude: float = 0.05,
    progress_callback: Optional[Callable[[float, int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict:
    if not network_inp_content:
        raise LeakageGenerationError("networkInpContent is empty")

    step_distance = float(delta_x_leak)
    if step_distance <= 0:
        raise LeakageGenerationError("deltaXLeak must be greater than 0")

    try:
        no_leak_portion_value = float(no_leak_portion)
    except (TypeError, ValueError):
        raise LeakageGenerationError("noLeakPortion must be numeric")

    if no_leak_portion_value < 0 or no_leak_portion_value > 1:
        raise LeakageGenerationError("noLeakPortion must be between 0 and 1")

    print("Data generation started")

    start_time = time.perf_counter()
    demand_values_lps = _parse_demand_range(leak_demand_range)
    normalized_attachments = _normalize_attachments(step3_sensor_attachments)
    polygons = _prepare_hexagons(hexagon_geojson)

    wn_template = _load_wn_from_inp(network_inp_content)
    pipe_paths_source_crs = _build_pipe_paths_source_crs(wn_template)
    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    dataset = []
    failures = []
    run_idx = 0

    # Pre-calculate total simulation runs for progress reporting
    total_runs = 0
    for pipe_name in wn_template.pipe_name_list:
        tp = wn_template.get_link(pipe_name)
        positions_count = len(_pipe_positions(float(tp.length), step_distance))
        if positions_count > 0 and pipe_paths_source_crs.get(str(pipe_name)):
            total_runs += len(demand_values_lps) * positions_count

    print("total runs: ", total_runs)
    logging.info(f"Total runs: {total_runs}")

    for pipe_name in wn_template.pipe_name_list:
        template_pipe = wn_template.get_link(pipe_name)
        positions = _pipe_positions(float(template_pipe.length), step_distance)
        if not positions:
            continue

        path_points = pipe_paths_source_crs.get(str(pipe_name))
        if not path_points:
            failures.append(
                {
                    "pipeId": pipe_name,
                    "error": "Missing source geometry for pipe path",
                }
            )
            continue

        for leak_demand_lps in demand_values_lps:
            leak_demand_m3s = leak_demand_lps / 1000.0
            for position in positions:
                run_idx += 1
                if (run_idx%50 == 0):
                    print(f"progress: {run_idx}/{total_runs}")
                if cancel_check is not None and cancel_check():
                    raise LeakageGenerationError("Task cancelled by user")
                if progress_callback is not None:
                    # Cap at 95% — reserve last 5% for no-leak phase
                    pct = (run_idx / total_runs * 95) if total_runs > 0 else 0
                    progress_callback(pct, run_idx, total_runs, f"Simulation {run_idx}/{total_runs}")
                try:
                    wn = _load_wn_from_inp(network_inp_content)
                    wn, leak_junction_name = _split_pipe_with_leak(wn, pipe_name, position, run_idx)

                    leak_junction = wn.get_node(leak_junction_name)
                    leak_junction.add_demand(leak_demand_m3s, pattern_name=None)

                    sim = wntr.sim.EpanetSimulator(wn)
                    results = sim.run_sim(convergence_error=False)

                    fraction = position / float(template_pipe.length)
                    leak_x, leak_y = _point_along_polyline(path_points, fraction)
                    leak_lon, leak_lat = to_wgs84.transform(leak_x, leak_y)
                    hexagon_id = _find_hexagon_index(leak_lon, leak_lat, polygons)

                    time_s, sensors = _extract_sensor_pressures(results, normalized_attachments, noise_amplitude)

                    dataset.append(
                        {
                            "runId": run_idx,
                            "pipeId": pipe_name,
                            "leakDemandLps": leak_demand_lps,
                            "leakPositionOnPipe": position,
                            "leakLocationWgs84": {
                                "longitude": leak_lon,
                                "latitude": leak_lat,
                            },
                            "hexagonId": hexagon_id,
                            "time": time_s,
                            "sensors": sensors,
                        }
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "runId": run_idx,
                            "pipeId": pipe_name,
                            "leakDemandLps": leak_demand_lps,
                            "leakPositionOnPipe": position,
                            "error": str(exc),
                        }
                    )

    if progress_callback is not None:
        progress_callback(96, run_idx, total_runs, "Generating no-leak samples...")

    no_leak_target_count = int(round(len(dataset) * no_leak_portion_value))
    if no_leak_target_count > 0 and dataset:
        wn_no_leak = _load_wn_from_inp(network_inp_content)
        sim_no_leak = wntr.sim.EpanetSimulator(wn_no_leak)
        results_no_leak = sim_no_leak.run_sim(convergence_error=False)

        pressures_df = results_no_leak.node["pressure"]
        base_times = [float(t) for t in list(pressures_df.index)]
        base_pressures: Dict[str, List[float]] = {}
        for col in pressures_df.columns:
            base_pressures[str(col)] = [float(v) for v in pressures_df[col].tolist()]

        sampled_indices = np.random.choice(
            len(dataset),
            size=no_leak_target_count,
            replace=False,
        )

        for sampled_idx in sampled_indices.tolist():
            template_run = dataset[int(sampled_idx)]
            run_idx += 1

            time_s, sensors = _apply_noise_to_base_pressures(
                base_pressures, base_times, normalized_attachments, noise_amplitude
            )

            dataset.append(
                {
                    "runId": run_idx,
                    "pipeId": -1,
                    "leakDemandLps": 0.0,
                    "leakPositionOnPipe": 0.0,
                    "leakLocationWgs84": None,
                    "hexagonId": -1,
                    "time": time_s,
                    "sensors": sensors,
                }
            )

    print("Data generation finished")
    end_time = time.perf_counter()

    logging.info(f"training data set generated successfully, time taken: {end_time - start_time}")
    return {
        "summary": {
            "totalRuns": run_idx,
            "successfulRuns": len(dataset),
            "failedRuns": len(failures),
            "stepDistance": step_distance,
            "leakDemandValuesLps": demand_values_lps,
        },
        "dataset": dataset,
        "failures": failures,
    }


def generate_random_test_data(
    *,
    hexagon_geojson: Dict,
    network_inp_content: str,
    crs: str,
    delta_x_leak: float,
    hexagon_radius: float,
    step3_sensor_attachments: List[Dict],
    num_samples: int = 20,
    leak_demand_min: float = 0.1,
    leak_demand_max: float = 2.0,
    noise_amplitude: float = 0.05,
    no_leak_portion: float = 0.1,
    progress_callback: Optional[Callable[[float, int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict:
    """Generate a random test dataset by placing leaks at random pipes/positions/demands.

    Unlike ``generate_leakage_dataset`` which exhaustively iterates over all pipes,
    this function randomly selects ``num_samples`` leak scenarios for testing purposes.
    """
    if not network_inp_content:
        raise LeakageGenerationError("networkInpContent is empty")

    step_distance = float(delta_x_leak)
    if step_distance <= 0:
        raise LeakageGenerationError("deltaXLeak must be greater than 0")

    if num_samples < 1:
        raise LeakageGenerationError("numSamples must be at least 1")

    if leak_demand_max < leak_demand_min:
        raise LeakageGenerationError("leakDemandMax must be >= leakDemandMin")

    normalized_attachments = _normalize_attachments(step3_sensor_attachments)
    polygons = _prepare_hexagons(hexagon_geojson)

    wn_template = _load_wn_from_inp(network_inp_content)
    pipe_paths_source_crs = _build_pipe_paths_source_crs(wn_template)
    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    # Build candidate list: pipes with valid geometry and positions
    candidates: List[Dict] = []
    sample = 0
    for pipe_name in wn_template.pipe_name_list:
        template_pipe = wn_template.get_link(pipe_name)
        positions = _pipe_positions(float(template_pipe.length), step_distance)
        if not positions:
            continue
        path_points = pipe_paths_source_crs.get(str(pipe_name))
        if not path_points:
            continue
        for position in positions:
            
            candidates.append({
                "pipeName": str(pipe_name),
                "position": position,
                "pipeLength": float(template_pipe.length),
                "pathPoints": path_points,
            })

    if not candidates:
        raise LeakageGenerationError("No valid pipe positions available for test data generation")

    dataset = []
    failures = []

    for run_idx in range(1, num_samples + 1):
        if cancel_check is not None and cancel_check():
            raise LeakageGenerationError("Task cancelled by user")

        if progress_callback is not None:
            pct = (run_idx / num_samples) * 95
            progress_callback(pct, run_idx, num_samples, f"Test sample {run_idx}/{num_samples}")
        sample+=1
        if (sample%50 == 0):
            print(f"progress: {sample}/{num_samples}")
        # Randomly select a pipe/position and demand
        candidate = random.choice(candidates)
        pipe_name = candidate["pipeName"]
        position = candidate["position"]
        pipe_length = candidate["pipeLength"]
        path_points = candidate["pathPoints"]
        leak_demand_lps = round(random.uniform(leak_demand_min, leak_demand_max), 4)
        leak_demand_m3s = leak_demand_lps / 1000.0

        try:
            wn = _load_wn_from_inp(network_inp_content)
            wn, leak_junction_name = _split_pipe_with_leak(wn, pipe_name, position, run_idx)

            leak_junction = wn.get_node(leak_junction_name)
            leak_junction.add_demand(leak_demand_m3s, pattern_name=None)

            sim = wntr.sim.EpanetSimulator(wn)
            results = sim.run_sim(convergence_error=False)

            fraction = position / pipe_length
            leak_x, leak_y = _point_along_polyline(path_points, fraction)
            leak_lon, leak_lat = to_wgs84.transform(leak_x, leak_y)
            hexagon_id = _find_hexagon_index(leak_lon, leak_lat, polygons)

            time_s, sensors = _extract_sensor_pressures(results, normalized_attachments, noise_amplitude)

            dataset.append({
                "runId": run_idx,
                "pipeId": pipe_name,
                "leakDemandLps": leak_demand_lps,
                "leakPositionOnPipe": position,
                "leakLocationWgs84": {
                    "longitude": leak_lon,
                    "latitude": leak_lat,
                },
                "hexagonId": hexagon_id,
                "time": time_s,
                "sensors": sensors,
            })
        except Exception as exc:
            print(f"failed run={run_idx} pipe={pipe_name} error={exc}", flush=True)
            failures.append({
                "runId": run_idx,
                "pipeId": pipe_name,
                "leakDemandLps": leak_demand_lps,
                "leakPositionOnPipe": position,
                "error": str(exc),
            })

    # Generate no-leak samples (unmodified network + noise only, no pipe splitting)
    no_leak_target_count = int(round(len(dataset) * no_leak_portion))
    if no_leak_target_count > 0 and dataset:
        if progress_callback is not None:
            progress_callback(96, num_samples, num_samples, "Generating no-leak samples...")

        wn_no_leak = _load_wn_from_inp(network_inp_content)
        sim_no_leak = wntr.sim.EpanetSimulator(wn_no_leak)
        results_no_leak = sim_no_leak.run_sim(convergence_error=False)

        pressures_df = results_no_leak.node["pressure"]
        base_times = [float(t) for t in list(pressures_df.index)]
        base_pressures: Dict[str, List[float]] = {}
        for col in pressures_df.columns:
            base_pressures[str(col)] = [float(v) for v in pressures_df[col].tolist()]

        for _ in range(no_leak_target_count):
            run_idx += 1
            time_s, sensors = _apply_noise_to_base_pressures(
                base_pressures, base_times, normalized_attachments, noise_amplitude
            )
            dataset.append({
                "runId": run_idx,
                "pipeId": -1,
                "leakDemandLps": 0.0,
                "leakPositionOnPipe": 0.0,
                "leakLocationWgs84": None,
                "hexagonId": -1,
                "time": time_s,
                "sensors": sensors,
            })

    total_runs = num_samples + no_leak_target_count
    return {
        "summary": {
            "totalRuns": total_runs,
            "successfulRuns": len(dataset),
            "failedRuns": len(failures),
        },
        "dataset": dataset,
        "failures": failures,
    }


def generate_leakage_dataset_refactored(
    *,
    hexagon_geojson: Dict,
    network_inp_content: str,
    crs: str,
    delta_x_leak: float,
    hexagon_radius: float,
    step3_sensor_attachments: List[Dict],
    leak_demand_range: List[float],
    no_leak_portion: float = 0.0,
    noise_amplitude: float = 0.05,
    progress_callback: Optional[Callable[[float, int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict:
    if not network_inp_content:
        raise LeakageGenerationError("networkInpContent is empty")

    step_distance = float(delta_x_leak)
    if step_distance <= 0:
        raise LeakageGenerationError("deltaXLeak must be greater than 0")

    try:
        no_leak_portion_value = float(no_leak_portion)
    except (TypeError, ValueError):
        raise LeakageGenerationError("noLeakPortion must be numeric")

    if no_leak_portion_value < 0 or no_leak_portion_value > 1:
        raise LeakageGenerationError("noLeakPortion must be between 0 and 1")

    print("Data generation started")

    start_time = time.perf_counter()
    demand_values_lps = _parse_demand_range(leak_demand_range)
    normalized_attachments = _normalize_attachments(step3_sensor_attachments)
    polygons = _prepare_hexagons(hexagon_geojson)

    wn_template = _load_wn_from_inp(network_inp_content)
    pipe_paths_source_crs = _build_pipe_paths_source_crs(wn_template)
    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    dataset = []
    failures = []
    run_idx = 0

    # Pre-calculate total simulation runs for progress reporting
    total_runs = 0
    for pipe_name in wn_template.pipe_name_list:
        tp = wn_template.get_link(pipe_name)
        positions_count = len(_pipe_positions(float(tp.length), step_distance))
        if positions_count > 0 and pipe_paths_source_crs.get(str(pipe_name)):
            total_runs += len(demand_values_lps) * positions_count

    print("total runs: ", total_runs)
    logging.info(f"Total runs: {total_runs}")


    for leak_demand_lps in demand_values_lps:
        print(f"Leak demand: {leak_demand_lps}")
        leak_demand_m3s = leak_demand_lps / 1000.0
        per_leak = []
        per_leak_failure = []
        runs = 0
        for pipe_name in wn_template.pipe_name_list:
            template_pipe = wn_template.get_link(pipe_name)
            positions = _pipe_positions(float(template_pipe.length), step_distance)
            if not positions:
                continue

            path_points = pipe_paths_source_crs.get(str(pipe_name))
            if not path_points:
                failures.append(
                    {
                        "pipeId": pipe_name,
                        "error": "Missing source geometry for pipe path",
                    }
                )
                continue

            for position in positions:
                run_idx += 1
                runs+=1
                if (run_idx%50 == 0):
                    print(f"progress: {run_idx}/{total_runs}")
                if cancel_check is not None and cancel_check():
                    raise LeakageGenerationError("Task cancelled by user")
                if progress_callback is not None:
                    # Cap at 95% — reserve last 5% for no-leak phase
                    pct = (run_idx / total_runs * 95) if total_runs > 0 else 0
                    progress_callback(pct, run_idx, total_runs, f"Simulation {run_idx}/{total_runs}")
                try:
                    wn = _load_wn_from_inp(network_inp_content)
                    wn, leak_junction_name = _split_pipe_with_leak(wn, pipe_name, position, run_idx)

                    leak_junction = wn.get_node(leak_junction_name)
                    leak_junction.add_demand(leak_demand_m3s, pattern_name=None)

                    sim = wntr.sim.EpanetSimulator(wn)
                    results = sim.run_sim(convergence_error=False)

                    fraction = position / float(template_pipe.length)
                    leak_x, leak_y = _point_along_polyline(path_points, fraction)
                    leak_lon, leak_lat = to_wgs84.transform(leak_x, leak_y)
                    hexagon_id = _find_hexagon_index(leak_lon, leak_lat, polygons)

                    time_s, sensors = _extract_sensor_pressures(results, normalized_attachments, noise_amplitude)
                    per_leak.append(
                        {
                            "runId": run_idx,
                            "pipeId": pipe_name,
                            "leakDemandLps": leak_demand_lps,
                            "leakPositionOnPipe": position,
                            "leakLocationWgs84": {
                                "longitude": leak_lon,
                                "latitude": leak_lat,
                            },
                            "hexagonId": hexagon_id,
                            "time": time_s,
                            "sensors": sensors,
                        }
                    )
                    dataset.append(
                        {
                            "runId": run_idx,
                            "pipeId": pipe_name,
                            "leakDemandLps": leak_demand_lps,
                            "leakPositionOnPipe": position,
                            "leakLocationWgs84": {
                                "longitude": leak_lon,
                                "latitude": leak_lat,
                            },
                            "hexagonId": hexagon_id,
                            "time": time_s,
                            "sensors": sensors,
                        }
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "runId": run_idx,
                            "pipeId": pipe_name,
                            "leakDemandLps": leak_demand_lps,
                            "leakPositionOnPipe": position,
                            "error": str(exc),
                        }
                    )
                    per_leak_failure.append(
                        {
                            "runId": run_idx,
                            "pipeId": pipe_name,
                            "leakDemandLps": leak_demand_lps,
                            "leakPositionOnPipe": position,
                            "error": str(exc),
                        }
                    )
        save_dicts({
            "summary": {
                "totalRuns": runs,
                "successfulRuns": len(per_leak),
                "failedRuns": len(per_leak_failure),
                "stepDistance": step_distance,
                "leakDemandValuesLps": demand_values_lps,
            },
            "dataset": per_leak,
            "failures": per_leak_failure,
        }, f"./data/leaks/{float(leak_demand_lps):.4f}.json")
        
    if progress_callback is not None:
        progress_callback(96, run_idx, total_runs, "Generating no-leak samples...")

    no_leak_target_count = int(round(len(dataset) * no_leak_portion_value))
    if no_leak_target_count > 0 and dataset:
        wn_no_leak = _load_wn_from_inp(network_inp_content)
        sim_no_leak = wntr.sim.EpanetSimulator(wn_no_leak)
        results_no_leak = sim_no_leak.run_sim(convergence_error=False)

        pressures_df = results_no_leak.node["pressure"]
        base_times = [float(t) for t in list(pressures_df.index)]
        base_pressures: Dict[str, List[float]] = {}
        for col in pressures_df.columns:
            base_pressures[str(col)] = [float(v) for v in pressures_df[col].tolist()]

        sampled_indices = np.random.choice(
            len(dataset),
            size=no_leak_target_count,
            replace=False,
        )

        for sampled_idx in sampled_indices.tolist():
            template_run = dataset[int(sampled_idx)]
            run_idx += 1

            time_s, sensors = _apply_noise_to_base_pressures(
                base_pressures, base_times, normalized_attachments, noise_amplitude
            )

            dataset.append(
                {
                    "runId": run_idx,
                    "pipeId": -1,
                    "leakDemandLps": 0.0,
                    "leakPositionOnPipe": 0.0,
                    "leakLocationWgs84": None,
                    "hexagonId": -1,
                    "time": time_s,
                    "sensors": sensors,
                }
            )

    print("Data generation finished")
    end_time = time.perf_counter()

    logging.info(f"training data set generated successfully, time taken: {end_time - start_time}")
    return {
        "summary": {
            "totalRuns": run_idx,
            "successfulRuns": len(dataset),
            "failedRuns": len(failures),
            "stepDistance": step_distance,
            "leakDemandValuesLps": demand_values_lps,
        },
        "dataset": dataset,
        "failures": failures,
    }