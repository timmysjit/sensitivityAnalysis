import json
from config.config import *
if CONFIG["PARALLEL"]:
    from leakgen.generator_parallel import *
else: 
    from leakgen.generator import *
from model_training import *
import time
from leakgen.utils import *
import logging

def main():
    id = save_init()
    start_time = time.perf_counter()
    file = open(CONFIG["GEOJSON"], "r")
    data = json.load(file)
    file.close()

    logging.info("saving config file")
    save_config(id)

    hexagon_geojson = json.loads(data["hexagonGeoJson"])
    network_inp_content = data["inpContent"]
    crs = data["crsCode"]
    hexagon_radius = data["hexagonRadius"]
    step3_sensor_attachments = data["step3SelectedNodeIds"]

    logging.info("dataset generation started")
    train_data = generate_leakage_dataset(
    hexagon_geojson = hexagon_geojson,
    network_inp_content = network_inp_content,
    crs = crs,
    delta_x_leak = CONFIG["DELTA_X_LEAK"],
    hexagon_radius = hexagon_radius,
    step3_sensor_attachments = step3_sensor_attachments,
    leak_demand_range = CONFIG["TRAIN"]["demand_range"],
    no_leak_portion = CONFIG["NO_LEAK_PORTION"],
    noise_amplitude = CONFIG["TRAIN"]["noise_amplitude"],
    )

    end_time = time.perf_counter()
    logging.info(f"time taken : {end_time - start_time}")
    
    if (CONFIG["TRAIN"]["save"]):
        file = open(f"./data/run{id}/train_data.json", "w")
        json.dump(train_data, file, indent = 4)
        file.close()
        logging.info("train leakage dataset saved")


if __name__ == "__main__":
    main()
