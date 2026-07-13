import os
import tempfile
from typing import Dict, List, Tuple
import wntr
from config.config import *
import json

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

def _pipe_positions(pipe_length: float, step_distance: float) -> List[float]:
    if pipe_length <= 0 or step_distance <= 0:
        return []

    positions = []
    current = step_distance
    while current < pipe_length:
        positions.append(current)
        current += step_distance

    return positions


def calc(*,
    network_inp_content: str,
    delta_x_leak: float,
    leak_demand_range: List[float],
    no_leak_portion: float = 0.0
):
    step_distance = float(delta_x_leak)
    demand_values_lps = _parse_demand_range(leak_demand_range)
    wn_template = _load_wn_from_inp(network_inp_content)
    pipe_paths_source_crs = _build_pipe_paths_source_crs(wn_template)
    
    total_runs = 0
    for pipe_name in wn_template.pipe_name_list:
        tp = wn_template.get_link(pipe_name)
        positions_count = len(_pipe_positions(float(tp.length), step_distance))
        if positions_count > 0 and pipe_paths_source_crs.get(str(pipe_name)):
            total_runs += len(demand_values_lps) * positions_count

    no_leak_target_count = int(total_runs * float(no_leak_portion))

    total = total_runs + no_leak_target_count
    print(f"leaks : {total_runs} , no leaks : {no_leak_target_count}, total : {total}")

def main():
    file = open(CONFIG["GEOJSON"], "r")
    data = json.load(file)
    file.close()
    network_inp_content = data["inpContent"]
    calc(network_inp_content= network_inp_content, delta_x_leak = CONFIG["DELTA_X_LEAK"], leak_demand_range= CONFIG["TRAIN"]["demand_range"], no_leak_portion = CONFIG["NO_LEAK_PORTION"])

if __name__ == "__main__":
    main()