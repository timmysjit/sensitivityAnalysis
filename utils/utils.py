import os
import json
from typing import Dict, List
from config.config import *
import logging
from datetime import datetime

def save_init():
    os.makedirs(f"./data", exist_ok= True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dir_path = f"./data/simulation_run_{timestamp}"
    os.makedirs(dir_path, exist_ok= True)
    logging.basicConfig(
        filename=f"{dir_path}/output.log", 
        filemode='w', # 'a' to append, 'w' to overwrite
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    return dir_path

def save_config(dir_path):
    try:
        file = open(f"{dir_path}/config.json", "w")
        json.dump(CONFIG, file, indent = 4)
        file.close()
        logging.info("successfully saved config file")
    except:
        logging.error("error saving config file")


def save(prediction, dir_path):
    try:
        file = open(f"{dir_path}/prediction.json", "w")
        json.dump(prediction, file, indent = 4)
        file.close()
        logging.info("successfully saved prediction output")
    except:
        logging.error("error saving prediction output")


def save_dicts(data, output_path):
    with open(output_path, 'w') as f:
        f.write(json.dumps(data, separators=(',', ':')) + '\n')

def relabel_sensor_placement(attachments: List[Dict], sensors: List[Dict]) -> List[Dict]:
    sensor_id_lookup = {att["nodeId"]: att["sensorId"] for att in attachments}

    new_sensors = []
    for sensor in sensors:
        new_sensors.append(
            {
                "sensorId": sensor_id_lookup.get(sensor["nodeId"]),
                "nodeId": str(sensor["nodeId"]),
                "pressure": sensor["pressure"],
            }
        )

    return new_sensors

def load_dataset_from_dir(directory : str , item_list : list[str]) -> Dict:
    dataset = []
    failures = []

    demand_values_lps = []
    step_distance = None
    next_run_id = 1

    totalRuns = 0
    successfulRuns = 0
    failedRuns = 0
    for item in item_list:
        print(item)
        item_path = os.path.join(directory, item)
        file = open(item_path, 'r')
        data = json.load(file)
        file.close()

        data = reconcile_run_id(data, start_run_id=next_run_id)

        summary = data.get("summary", {})
        dataset.extend(data.get("dataset", []))
        failures.extend(data.get("failures", []))
        totalRuns += summary.get("totalRuns", 0)
        successfulRuns += summary.get("successfulRuns", 0)
        failedRuns += summary.get("failedRuns", 0)
        next_run_id += summary.get("totalRuns", 0)
        
        if step_distance is None:
            step_distance = summary.get("stepDistance")

        for value in summary.get("leakDemandValuesLps", []):
            if value not in demand_values_lps:
                demand_values_lps.append(value)

    return {
        "summary": {
            "totalRuns": totalRuns,
            "successfulRuns": successfulRuns,
            "failedRuns": failedRuns,
            "stepDistance": step_distance,
            "leakDemandValuesLps": sorted(demand_values_lps),
        },
        "dataset": dataset,
        "failures": failures
    }

def reconcile_run_id(payload : Dict, start_run_id: int = 1) -> Dict:
    dataset = payload.get("dataset", [])
    failures = payload.get("failures", [])

    indexed_runs = []
    for run in dataset:
        if isinstance(run, dict):
            indexed_runs.append((int(run.get("runId", 0)), run))

    for run in failures:
        if isinstance(run, dict):
            indexed_runs.append((int(run.get("runId", 0)), run))

    indexed_runs.sort(key=lambda item: item[0])

    next_id = start_run_id
    for _, run in indexed_runs:
        run["runId"] = next_id
        next_id += 1

    summary = payload.setdefault("summary", {})
    summary["successfulRuns"] = len(dataset)
    summary["failedRuns"] = len(failures)
    summary["totalRuns"] = len(dataset) + len(failures)

    return payload


'''
Optimal sensor placement algorithm
'''
def pagerank_sensor_placement(payload :Dict, number_of_sensors : int) -> List[str]:
    if number_of_sensors <= 0:
        return []

    network_geojson = payload.get("networkGeoJson", payload)
    if isinstance(network_geojson, str):
        network_geojson = json.loads(network_geojson)

    if not isinstance(network_geojson, dict):
        raise ValueError("payload must contain a valid networkGeoJson object or JSON string")

    features = network_geojson.get("features", [])
    if not isinstance(features, list):
        raise ValueError("networkGeoJson.features must be a list")

    junction_ids = []
    junction_set = set()
    for feature in features:
        if not isinstance(feature, dict):
            continue
        properties = feature.get("properties", {})
        if properties.get("type") != "Node" or properties.get("category") != "Junction":
            continue

        node_id = str(properties.get("id", "")).strip()
        if node_id and node_id not in junction_set:
            junction_ids.append(node_id)
            junction_set.add(node_id)

    if not junction_ids:
        return []

    adjacency = {node_id: set() for node_id in junction_ids}
    for feature in features:
        if not isinstance(feature, dict):
            continue
        properties = feature.get("properties", {})
        if properties.get("type") != "Link":
            continue

        us_node_id = str(properties.get("usNodeId", "")).strip()
        ds_node_id = str(properties.get("dsNodeId", "")).strip()
        if us_node_id in junction_set and ds_node_id in junction_set and us_node_id != ds_node_id:
            adjacency[us_node_id].add(ds_node_id)
            adjacency[ds_node_id].add(us_node_id)

    node_count = len(junction_ids)
    alpha = float(payload.get("pagerankAlpha", 0.85))
    tolerance = float(payload.get("pagerankTolerance", 1e-8))
    max_iterations = int(payload.get("pagerankMaxIterations", 2000))

    scores = {node_id: 1.0 / node_count for node_id in junction_ids}
    teleport = 1.0 / node_count

    for _ in range(max_iterations):
        dangling_mass = sum(scores[node_id] for node_id in junction_ids if len(adjacency[node_id]) == 0)
        new_scores = {}

        for node_id in junction_ids:
            score_sum = dangling_mass / node_count
            for neighbor_id in adjacency[node_id]:
                neighbor_degree = len(adjacency[neighbor_id])
                if neighbor_degree > 0:
                    score_sum += scores[neighbor_id] / neighbor_degree

            new_scores[node_id] = alpha * score_sum + (1.0 - alpha) * teleport

        diff = sum((new_scores[node_id] - scores[node_id]) ** 2 for node_id in junction_ids) ** 0.5
        scores = new_scores
        if diff <= tolerance:
            break

    ranked_nodes = sorted(junction_ids, key=lambda node_id: (scores[node_id], node_id))
    return ranked_nodes[:min(number_of_sensors, len(ranked_nodes))]