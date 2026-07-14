# pyright: reportMissingImports=false

import json
import os
import uuid
from typing import Any, Callable, Dict, List, Optional

import torch
from torch_geometric.loader import DataLoader

from .graph import create_graph, load_wn_from_inp_content
from .model import GCN



MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Learning rates
PRETRAINING_LR: float = 1e-4
FINETUNING_LR: float = 1e-5

# Number of CPU cores to use for training (DataLoader workers & PyTorch threads).
# Set to None to use all available cores.
TRAINING_NUM_CORES: Optional[int] = 1

# Configure CPU parallelism BEFORE any torch operations to avoid
# "cannot set number of interop threads after parallel work has started" error.
_available_cores = os.cpu_count() or 1
_cores = min(TRAINING_NUM_CORES, _available_cores) if TRAINING_NUM_CORES and TRAINING_NUM_CORES > 0 else _available_cores
torch.set_num_threads(_cores)
torch.set_num_interop_threads(_cores)

# Detect GPU availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelTrainingError(Exception):
    pass


def _model_path(model_id: str) -> str:
    safe_id = os.path.basename(str(model_id))
    return os.path.join(MODELS_DIR, f'{safe_id}.pt')


def _meta_path(model_id: str) -> str:
    safe_id = os.path.basename(str(model_id))
    return os.path.join(MODELS_DIR, f'{safe_id}.meta.json')


def _find_finetuned_model(base_model_id: str) -> Optional[str]:
    """Find the most recent fine-tuned model derived from base_model_id.

    Scans the models directory for .meta.json files whose baseModelId
    matches the given ID. Returns the fine-tuned model's ID, or None.
    """
    best_id: Optional[str] = None
    best_mtime: float = 0.0

    for filename in os.listdir(MODELS_DIR):
        if not filename.endswith('.meta.json'):
            continue
        filepath = os.path.join(MODELS_DIR, filename)
        try:
            with open(filepath, 'r') as f:
                meta = json.load(f)
            if meta.get('baseModelId') == base_model_id:
                mtime = os.path.getmtime(filepath)
                if mtime > best_mtime:
                    best_mtime = mtime
                    best_id = meta.get('modelId')
        except (json.JSONDecodeError, OSError, KeyError):
            continue

    return best_id



def train_leakage_model(
    *,
    dataset_payload: Dict[str, Any],
    network_inp_content: str,
    epoch_count: int = 1,
    all_hexagon_ids: Optional[List[int]] = None,
    selected_node_ids: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[int, int, int, int, float], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    if not isinstance(dataset_payload, dict):
        raise ModelTrainingError("leakageDataset must be an object")

    dataset = dataset_payload.get("dataset")
    if not isinstance(dataset, list) or not dataset:
        raise ModelTrainingError("leakageDataset.dataset must be a non-empty array")

    if not network_inp_content or not str(network_inp_content).strip():
        raise ModelTrainingError("networkInpContent is required")

    if not isinstance(epoch_count, int) or epoch_count < 1:
        raise ModelTrainingError("epoch_count must be an integer greater than or equal to 1")

    # Use pre-configured core count for DataLoader workers
    dataloader_workers = min(_cores, 4)  # cap workers to avoid overhead

    # Determine the full set of hexagon IDs.
    # Priority: explicit parameter > dataset-level field > dataset-derived
    if all_hexagon_ids and len(all_hexagon_ids) > 0:
        hexagon_ids = sorted(set(int(h) for h in all_hexagon_ids))
    elif isinstance(dataset_payload.get("allHexagonIds"), list) and dataset_payload["allHexagonIds"]:
        hexagon_ids = sorted(set(int(h) for h in dataset_payload["allHexagonIds"]))
    else:
        hexagon_ids = sorted(
            {
                int(run.get("hexagonId"))
                for run in dataset
                if isinstance(run, dict)
                and run.get("hexagonId") is not None
                and int(run.get("hexagonId")) >= 0
            }
        )
    if not hexagon_ids:
        raise ModelTrainingError("No valid hexagonId labels found in leakageDataset.dataset")

    hex_to_class_index = {hex_id: idx for idx, hex_id in enumerate(hexagon_ids)}

    try:
        wn = load_wn_from_inp_content(network_inp_content)
    except Exception as exc:
        raise ModelTrainingError(f"Failed to parse networkInpContent: {exc}")

    try:
        data_list = create_graph(dataset, wn, hex_to_class_index, selected_node_ids=selected_node_ids)
    except Exception as exc:
        raise ModelTrainingError(f"Failed to create graph dataset: {exc}")

    if not data_list:
        raise ModelTrainingError("No valid training samples after graph preparation")

    num_node_features = int(data_list[0].x.shape[1])
    num_hexagons = len(hexagon_ids)

    model = GCN(num_node_features=num_node_features, num_hexagons=num_hexagons).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=PRETRAINING_LR)
    criterion = torch.nn.CrossEntropyLoss()
    loader = DataLoader(data_list, batch_size=min(16, len(data_list)), shuffle=True, num_workers=dataloader_workers)

    print(f'[train-model] Using device: {DEVICE}')
    model.train()
    train_losses = []

    for epoch in range(epoch_count):
        total_loss = 0.0
        total_batches = 0
        print(f"epochs: {epoch}/{epoch_count}")

        for batch in loader:
            if cancel_check is not None and cancel_check():
                raise ModelTrainingError("Task cancelled by user")
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch.x, batch.edge_index, batch.batch)
            target = batch.y_hex.view(-1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

            if progress_callback is not None:
                progress_callback(epoch + 1, epoch_count, total_batches, len(loader), float(loss.item()))

        average_loss = total_loss / total_batches if total_batches > 0 else 0.0
        train_losses.append(average_loss)

    final_loss = train_losses[-1] if train_losses else 0.0

    # Persist model weights and metadata to disk (save on CPU for portability)
    model_id = str(uuid.uuid4())
    model.cpu()
    torch.save(model.state_dict(), _model_path(model_id))

    # Build reverse mapping: class index → hexagon id
    class_to_hex = {idx: hex_id for hex_id, idx in hex_to_class_index.items()}

    # Extract sensor node IDs: use selectedNodeIds if provided, otherwise all nodes from dataset
    if selected_node_ids:
        sensor_node_ids = list(selected_node_ids)
    else:
        sensor_node_ids: List[str] = []
        first_run = dataset[0] if dataset else {}
        sensors_list = first_run.get("sensors", [])
        for s in sensors_list:
            if isinstance(s, dict) and s.get("nodeId"):
                sensor_node_ids.append(str(s["nodeId"]))

    meta = {
        "modelId": model_id,
        "numNodeFeatures": num_node_features,
        "numHexagons": num_hexagons,
        "hexToClassIndex": {str(k): v for k, v in hex_to_class_index.items()},
        "classToHexId": {str(k): v for k, v in class_to_hex.items()},
        "sensorNodeIds": sensor_node_ids,
        "epochCount": epoch_count,
        "trainLoss": final_loss,
    }
    with open(_meta_path(model_id), 'w') as f:
        json.dump(meta, f)

    return {
        "status": "success",
        "message": "Step 6 training completed.",
        "modelId": model_id,
        "epochCount": epoch_count,
        "trainLoss": final_loss,
        "trainLosses": train_losses,
        "numSamples": len(data_list),
        "numClasses": num_hexagons,
    }


def predict_leakage(
    *,
    model_id: str,
    test_dataset: List[Dict[str, Any]],
    network_inp_content: str,
    selected_node_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load a saved model and run inference on test data.

    Automatically uses the latest fine-tuned model if one exists for the given model_id.
    Returns per-sample predictions with predicted hexagon, confidence, and actual hexagon.
    """
    # Auto-resolve to fine-tuned model if available
    finetuned_id = _find_finetuned_model(model_id)
    resolved_model_id = finetuned_id if finetuned_id else model_id

    meta_file = _meta_path(resolved_model_id)
    model_file = _model_path(resolved_model_id)

    if not os.path.isfile(meta_file):
        raise ModelTrainingError(f"Model metadata not found for modelId: {resolved_model_id}")
    if not os.path.isfile(model_file):
        raise ModelTrainingError(f"Model weights not found for modelId: {resolved_model_id}")

    with open(meta_file, 'r') as f:
        meta = json.load(f)

    num_node_features = int(meta["numNodeFeatures"])
    num_hexagons = int(meta["numHexagons"])
    hex_to_class_index = {int(k): int(v) for k, v in meta["hexToClassIndex"].items()}
    class_to_hex = {int(k): int(v) for k, v in meta["classToHexId"].items()}

    # Reconstruct the model and load weights
    model = GCN(num_node_features=num_node_features, num_hexagons=num_hexagons)
    model.load_state_dict(torch.load(model_file, weights_only=True, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Parse the network
    try:
        wn = load_wn_from_inp_content(network_inp_content)
    except Exception as exc:
        raise ModelTrainingError(f"Failed to parse networkInpContent: {exc}")

    # Build graph data for each test sample
    try:
        data_list = create_graph(test_dataset, wn, hex_to_class_index, selected_node_ids=selected_node_ids)
    except Exception as exc:
        raise ModelTrainingError(f"Failed to create graph dataset for prediction: {exc}")

    if not data_list:
        raise ModelTrainingError("No valid test samples after graph preparation")

    loader = DataLoader(data_list, batch_size=len(data_list), shuffle=False)

    predictions = []
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            output = model(batch.x, batch.edge_index, batch.batch)
            probs = torch.softmax(output, dim=1)
            confidences, predicted_classes = torch.max(probs, dim=1)
            actual_classes = batch.y_hex.view(-1)

            for i in range(len(predicted_classes)):
                pred_class = int(predicted_classes[i].item())
                actual_class = int(actual_classes[i].item())
                confidence = float(confidences[i].item())
                pred_hex = class_to_hex.get(pred_class, -1)
                actual_hex = class_to_hex.get(actual_class, -1)

                # Full probability distribution: {hexagonId: probability}
                sample_probs = probs[i]
                hex_probabilities = {}
                for cls_idx in range(sample_probs.shape[0]):
                    hex_id = class_to_hex.get(cls_idx, -1)
                    hex_probabilities[hex_id] = round(float(sample_probs[cls_idx].item()), 4)

                predictions.append({
                    "predictedHexagonId": pred_hex,
                    "actualHexagonId": actual_hex,
                    "confidence": round(confidence, 4),
                    "correct": pred_hex == actual_hex,
                    "hexagonProbabilities": hex_probabilities,
                })

                total += 1
                if pred_hex == actual_hex:
                    correct += 1

    accuracy = correct / total if total > 0 else 0.0

    return {
        "modelIdUsed": resolved_model_id,
        "fineTuned": finetuned_id is not None,
        "selectedNodeIdsUsed": list(selected_node_ids) if selected_node_ids else None,
        "predictions": predictions,
        "accuracy": round(accuracy, 4),
        "total": total,
        "correct": correct,
    }


def transfer_learning(
    *,
    model_id: str,
    dataset_payload: Dict[str, Any],
    network_inp_content: str,
    epoch_count: int = 1,
    learning_rate: float = FINETUNING_LR,
    selected_node_ids: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[int, int, int, int, float], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Feature-extraction transfer learning.

    Loads a pre-trained model by model_id, freezes the GCN (conv) layers,
    and fine-tunes only the classification head (linHexagon).
    """
    # Validate base model exists
    meta_file = _meta_path(model_id)
    model_file = _model_path(model_id)

    if not os.path.isfile(meta_file):
        raise ModelTrainingError(f"Model metadata not found for modelId: {model_id}")
    if not os.path.isfile(model_file):
        raise ModelTrainingError(f"Model weights not found for modelId: {model_id}")

    with open(meta_file, 'r') as f:
        meta = json.load(f)

    base_num_node_features = int(meta["numNodeFeatures"])
    base_num_hexagons = int(meta["numHexagons"])

    # Validate dataset
    if not isinstance(dataset_payload, dict):
        raise ModelTrainingError("leakageDataset must be an object")

    dataset = dataset_payload.get("dataset")
    if not isinstance(dataset, list) or not dataset:
        raise ModelTrainingError("leakageDataset.dataset must be a non-empty array")

    if not network_inp_content or not str(network_inp_content).strip():
        raise ModelTrainingError("networkInpContent is required")

    if not isinstance(epoch_count, int) or epoch_count < 1:
        raise ModelTrainingError("epoch_count must be an integer greater than or equal to 1")

    dataloader_workers = min(_cores, 4)

    # Use the same hexagon mapping from the base model
    hex_to_class_index = {int(k): int(v) for k, v in meta["hexToClassIndex"].items()}
    num_hexagons = base_num_hexagons

    # Parse network and build graph dataset
    try:
        wn = load_wn_from_inp_content(network_inp_content)
    except Exception as exc:
        raise ModelTrainingError(f"Failed to parse networkInpContent: {exc}")

    try:
        data_list = create_graph(dataset, wn, hex_to_class_index, selected_node_ids=selected_node_ids)
    except Exception as exc:
        raise ModelTrainingError(f"Failed to create graph dataset: {exc}")

    if not data_list:
        raise ModelTrainingError("No valid training samples after graph preparation")

    num_node_features = int(data_list[0].x.shape[1])

    # Validate feature compatibility with the base model
    if num_node_features != base_num_node_features:
        raise ModelTrainingError(
            f"Feature mismatch: base model expects {base_num_node_features} node features, "
            f"but new dataset has {num_node_features}"
        )

    # Load pre-trained model (use base model's num_hexagons to match saved weights)
    model = GCN(num_node_features=base_num_node_features, num_hexagons=base_num_hexagons)
    model.load_state_dict(torch.load(model_file, weights_only=True, map_location=DEVICE))
    model.to(DEVICE)

    # Freeze GCN (conv) layers — only train the classification head
    for param in model.convs.parameters():
        param.requires_grad = False

    head_params = list(model.linHexagon.parameters()) + list(model.linHexagon2.parameters())
    optimizer = torch.optim.Adam(head_params, lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    loader = DataLoader(data_list, batch_size=min(16, len(data_list)), shuffle=True, num_workers=dataloader_workers)

    print(f'[transfer-learning] Using device: {DEVICE}')
    print(f'[transfer-learning] Base model: {model_id}, epochs: {epoch_count}, lr: {learning_rate}')
    print(f'[transfer-learning] Frozen: convs | Trainable: linHexagon, linHexagon2')
    model.train()
    train_losses = []

    for epoch in range(epoch_count):
        total_loss = 0.0
        total_batches = 0

        for batch in loader:
            if cancel_check is not None and cancel_check():
                raise ModelTrainingError("Task cancelled by user")
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch.x, batch.edge_index, batch.batch)
            target = batch.y_hex.view(-1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

            if progress_callback is not None:
                progress_callback(epoch + 1, epoch_count, total_batches, len(loader), float(loss.item()))

        average_loss = total_loss / total_batches if total_batches > 0 else 0.0
        train_losses.append(average_loss)

    final_loss = train_losses[-1] if train_losses else 0.0

    # Save fine-tuned model with a new ID
    new_model_id = str(uuid.uuid4())
    model.cpu()
    torch.save(model.state_dict(), _model_path(new_model_id))

    # Build reverse mapping
    class_to_hex = {idx: hex_id for hex_id, idx in hex_to_class_index.items()}

    # Extract sensor node IDs: use selectedNodeIds if provided, otherwise all nodes from dataset
    if selected_node_ids:
        sensor_node_ids = list(selected_node_ids)
    else:
        sensor_node_ids: List[str] = []
        first_run = dataset[0] if dataset else {}
        sensors_list = first_run.get("sensors", [])
        for s in sensors_list:
            if isinstance(s, dict) and s.get("nodeId"):
                sensor_node_ids.append(str(s["nodeId"]))

    new_meta = {
        "modelId": new_model_id,
        "baseModelId": model_id,
        "numNodeFeatures": num_node_features,
        "numHexagons": num_hexagons,
        "hexToClassIndex": {str(k): v for k, v in hex_to_class_index.items()},
        "classToHexId": {str(k): v for k, v in class_to_hex.items()},
        "sensorNodeIds": sensor_node_ids,
        "epochCount": epoch_count,
        "learningRate": learning_rate,
        "trainLoss": final_loss,
    }
    with open(_meta_path(new_model_id), 'w') as f:
        json.dump(new_meta, f)

    return {
        "status": "success",
        "message": "Transfer learning (classification head fine-tuning) completed.",
        "modelId": new_model_id,
        "baseModelId": model_id,
        "epochCount": epoch_count,
        "trainLoss": final_loss,
        "trainLosses": train_losses,
        "numSamples": len(data_list),
        "numClasses": num_hexagons,
    }