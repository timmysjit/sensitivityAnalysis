import os
import glob
import re
import json
from typing import Any, Dict, Iterable, List, Optional, Union
from config.config import *
import logging

def save_init():
    os.makedirs(f"./data", exist_ok= True)
    runs = [x for x in os.listdir('./data')]
    os.makedirs(f"./data/run{len(runs)}", exist_ok= True)
    logging.basicConfig(
        filename=f"./data/run{len(runs)}/output.log", 
        filemode='w', # 'a' to append, 'w' to overwrite
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    return len(runs)

def save_config(id: int):
    try:
        file = open(f"./data/run{id}/config.json", "w")
        json.dump(CONFIG, file, indent = 4)
        file.close()
        logging.info("successfully saved config file")
    except:
        logging.error("error saving config file")


def save(prediction, id : int):
    try:
        file = open(f"./data/run{id}/prediction.json", "w")
        json.dump(prediction, file, indent = 4)
        file.close()
        logging.info("successfully saved prediction output")
    except:
        logging.error("error saving prediction output")


def save_dicts(data, output_path):
    with open(output_path, 'w') as f:
        f.write(json.dumps(data, separators=(',', ':')) + '\n')


# ---------------------------------------------------------------------------
# Loaders for training data saved in per-leak-demand-lps JSON batch files.
#
# Each batch file is expected to follow the same schema produced by the
# generator, e.g.
#   {"summary": {...}, "dataset": [ {..., "leakDemandLps": <float>, ...}, ... ],
#    "failures": [...], "allHexagonIds": [...]?}
# The filename is expected to contain the lps value, e.g.
#   train_data_lps_0.50.json, train_0.50.json, batch_0p50.json, ...
# The helpers here are permissive: any decimal number found in the basename
# is treated as the lps value; if none is present, the file is loaded
# without a numeric sort key.
# ---------------------------------------------------------------------------

_LPS_RE = re.compile(r"(\d+(?:[._]\d+)?)")


def _extract_lps_from_filename(path: str) -> Optional[float]:
    """Return the first decimal number found in the basename, or None."""
    name = os.path.splitext(os.path.basename(path))[0]
    match = _LPS_RE.search(name)
    if not match:
        return None
    try:
        return float(match.group(1).replace("_", "."))
    except ValueError:
        return None


def load_dataset_batch(path: str) -> Dict[str, Any]:
    """Load a single training-data JSON file and return the parsed dict."""
    with open(path, "r") as f:
        return json.load(f)


def load_dataset_batches(
    dir_or_paths: Union[str, Iterable[str]],
    pattern: str = "*.json",
    sort_by_lps: bool = True,
    recursive: bool = False,
) -> Dict[str, Any]:
    """Load and merge several per-lps training-data JSON files.

    Parameters
    ----------
    dir_or_paths : str | Iterable[str]
        Either a directory to scan or an explicit iterable of file paths.
    pattern : str
        Glob pattern used when ``dir_or_paths`` is a directory.
    sort_by_lps : bool
        When True, files are ordered by the numeric value parsed from their
        basename (files without a number are appended last).
    recursive : bool
        When True and a directory is given, search recursively.

    Returns
    -------
    dict
        A merged payload with the shape expected by ``train_leakage_model``:
        ``{"dataset": [...], "allHexagonIds": [...], "failures": [...],
           "summary": {...}}``.
    """
    if isinstance(dir_or_paths, str):
        if recursive:
            paths = sorted(glob.glob(os.path.join(dir_or_paths, "**", pattern),
                                     recursive=True))
        else:
            paths = sorted(glob.glob(os.path.join(dir_or_paths, pattern)))
    else:
        paths = list(dir_or_paths)

    if sort_by_lps:
        paths.sort(key=lambda p: (
            _extract_lps_from_filename(p) is None,
            _extract_lps_from_filename(p) or 0.0,
            p,
        ))

    merged_dataset: List[Dict[str, Any]] = []
    merged_failures: List[Any] = []
    merged_hex: set = set()
    batch_summaries: List[Dict[str, Any]] = []
    lps_values: set = set()
    next_run_id = 1

    for path in paths:
        payload = load_dataset_batch(path)
        entries = payload.get("dataset") or []

        # Re-number runIds across batches so they stay unique after merge.
        for entry in entries:
            entry = dict(entry)
            entry["runId"] = next_run_id
            next_run_id += 1
            lps = entry.get("leakDemandLps")
            if lps is not None:
                try:
                    lps_values.add(float(lps))
                except (TypeError, ValueError):
                    pass
            merged_dataset.append(entry)

        for h in (payload.get("allHexagonIds") or []):
            try:
                merged_hex.add(int(h))
            except (TypeError, ValueError):
                continue

        merged_failures.extend(payload.get("failures") or [])

        batch_summaries.append({
            "file": os.path.basename(path),
            "lps": _extract_lps_from_filename(path),
            "num_samples": len(entries),
            "summary": payload.get("summary"),
        })

    if not merged_hex:
        for entry in merged_dataset:
            h = entry.get("hexagonId")
            if h is None or h == -1:
                continue
            try:
                merged_hex.add(int(h))
            except (TypeError, ValueError):
                continue

    return {
        "dataset": merged_dataset,
        "allHexagonIds": sorted(merged_hex),
        "failures": merged_failures,
        "summary": {
            "num_batches": len(paths),
            "num_samples": len(merged_dataset),
            "leak_demand_values_lps": sorted(lps_values),
            "batches": batch_summaries,
        },
    }


def load_run(run_id: int,
             data_root: str = "./data",
             pattern: str = "train_data*.json") -> Dict[str, Any]:
    """Load all per-lps batch files for a given run directory.

    Convenience wrapper around :func:`load_dataset_batches` that targets
    ``{data_root}/run{run_id}/``.
    """
    run_dir = os.path.join(data_root, f"run{run_id}")
    return load_dataset_batches(run_dir, pattern=pattern)


