# pyright: reportMissingImports=false

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import FeaturePropagation
import wntr
import os
import tempfile


def load_wn_from_inp_content(network_inp_content: str):
    if not network_inp_content or not str(network_inp_content).strip():
        raise ValueError("networkInpContent is empty")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".inp", delete=False) as tmp:
            tmp.write(network_inp_content)
            temp_path = tmp.name
        return wntr.network.WaterNetworkModel(temp_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def create_graph(dataset_list, wn, hex_to_class_index, selected_node_ids=None):
    if not isinstance(dataset_list, list) or not dataset_list:
        raise ValueError("dataset_list must be a non-empty array")

    junction_names = list(getattr(wn, "junction_name_list", []))
    if not junction_names:
        raise ValueError("No junctions found in network")

    edge_index = create_edge_index(wn)
    node_to_index = {name: idx for idx, name in enumerate(junction_names)}

    # If selectedNodeIds is provided, only use those nodes as "known" sensors.
    # All other nodes will remain NaN and be filled by FeaturePropagation.
    allowed_node_ids = set(selected_node_ids) if selected_node_ids else None

    data_list = []
    for run in dataset_list:
        if not isinstance(run, dict):
            continue

        time_values = run.get("time")
        if not isinstance(time_values, list) or not time_values:
            continue

        num_nodes = len(junction_names)
        num_steps = len(time_values)

        # Unknown nodes remain NaN and are filled by feature propagation.
        x = torch.full((num_nodes, num_steps), float("nan"), dtype=torch.float)

        sensors = run.get("sensors") if isinstance(run.get("sensors"), list) else []
        for sensor in sensors:
            if not isinstance(sensor, dict):
                continue

            node_id = str(sensor.get("nodeId", "")).strip()

            # If filtering is active, skip nodes not in the selected set
            if allowed_node_ids is not None and node_id not in allowed_node_ids:
                continue

            node_index = node_to_index.get(node_id)
            if node_index is None:
                continue

            pressure_values = sensor.get("pressure")
            if not isinstance(pressure_values, list) or len(pressure_values) != num_steps:
                continue

            x[node_index] = torch.tensor(pressure_values, dtype=torch.float)

        hexagon_id = int(run.get("hexagonId", -1))
        if hexagon_id not in hex_to_class_index:
            continue

        y_hex = torch.tensor(hex_to_class_index[hexagon_id], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y_hex=y_hex)
        transform = FeaturePropagation(missing_mask=torch.isnan(data.x))
        data = transform(data)
        data_list.append(data)

    return data_list


def create_edge_index(wn):
    edge_index_list = [[], []]
    for _, link in wn.links():
        start_node = wn.get_node(link.start_node_name)
        end_node = wn.get_node(link.end_node_name)
        if isinstance(start_node, wntr.network.Junction) and isinstance(end_node, wntr.network.Junction):
            start_index = wn.junction_name_list.index(link.start_node_name)
            end_index = wn.junction_name_list.index(link.end_node_name)
            edge_index_list[0].append(start_index)
            edge_index_list[1].append(end_index)
            edge_index_list[0].append(end_index)
            edge_index_list[1].append(start_index)

    edge_index = torch.tensor(edge_index_list, dtype=torch.long)
    return edge_index