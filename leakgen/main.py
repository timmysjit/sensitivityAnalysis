import json
from leakgen.generator_parallel import *
from model_training import *
import time
from config.config import *
from leakgen.utils import *
import logging

if CONFIG["PARALLEL"]:
    from leakgen.generator_parallel import *
else:
    from leakgen.generator import *

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

    if (CONFIG["TRAIN"]["save"]):
        file = open(f"./data/run{id}/train_data.json", "w")
        json.dump(train_data, file)
        file.close()
        logging.info("train leakage dataset saved")

    print("training started")
    train_output = train_leakage_model(dataset_payload=train_data, network_inp_content=network_inp_content, epoch_count=CONFIG["EPOCH"])
    print("training finished")

    if (CONFIG["TRAIN"]["save"]):
        logging.info("saving train output")
        file = open(f"./data/run{id}/train.json", "w")
        json.dump(train_output, file, indent = 4)
        file.close()
        logging.info("train output saved")

    gcn_model_id = train_output["modelId"]

    print("transfer learning dataset generation started")
    tl_data = generate_random_test_data(
    hexagon_geojson = hexagon_geojson,
    network_inp_content = network_inp_content,
    crs = crs,
    delta_x_leak = CONFIG["DELTA_X_LEAK"],
    hexagon_radius = hexagon_radius,
    step3_sensor_attachments = step3_sensor_attachments,
    num_samples = CONFIG["TL"]["num_samples"],
    leak_demand_min = CONFIG["TL"]["leak_demand_min"],
    leak_demand_max = CONFIG["TL"]["leak_demand_max"],
    noise_amplitude = CONFIG["TL"]["noise_amplitude"],
    no_leak_portion = CONFIG["NO_LEAK_PORTION"],
    )
    print("transfer learning dataset generation finished")


    if (CONFIG["TL"]["save"]):
        logging.info("saving transfer learning dataset")
        file = open(f"./data/run{id}/tl_data.json", "w")
        json.dump(tl_data, file, indent = 4)
        file.close()
        logging.info("saved transfer learning dataset")

    print("transfer learning training started")
    tl_output = transfer_learning(model_id=gcn_model_id, dataset_payload=tl_data, network_inp_content=network_inp_content, epoch_count=CONFIG["EPOCH"])
    print("transfer learning training finished")
    tl_model_id = tl_output["modelId"]

    if (CONFIG["TL"]["save"]):
        logging.info("saving transfer learning output")
        file = open(f"./data/run{id}/tl_train.json", "w")
        json.dump(tl_output, file, indent = 4)
        file.close()
        logging.info("saved transfer learning output")
    
    print("test dataset generation started")
    test_data = generate_random_test_data(
        hexagon_geojson=hexagon_geojson,
        network_inp_content=network_inp_content,
        crs=crs,
        delta_x_leak=CONFIG["DELTA_X_LEAK"],
        hexagon_radius=hexagon_radius,
        step3_sensor_attachments=step3_sensor_attachments,
        num_samples=CONFIG["TEST"]["num_samples"],
        leak_demand_min=CONFIG["TEST"]["leak_demand_min"],
        leak_demand_max=CONFIG["TEST"]["leak_demand_max"],
        noise_amplitude=CONFIG["TEST"]["noise_amplitude"],
        no_leak_portion=CONFIG["NO_LEAK_PORTION"],
    )
    print("test dataset generation finished")

    if (CONFIG["TEST"]["save"]):
        logging.info("saving test dataset")
        file = open(f"./data/run{id}/test_data.json", "w")
        json.dump(test_data, file, indent = 4)
        file.close()
        logging.info("saved test dataset")
    
    print("prediction on test data started")
    prediction = predict_leakage(model_id=tl_model_id, test_dataset=test_data["dataset"], network_inp_content=network_inp_content)
    print("prediction on test data finished")

    logging.info("saving prediction output")
    save(prediction, id)
    end_time = time.perf_counter()

    print("Time taken: ", end_time - start_time)
    logging.info(f"Time taken: {end_time - start_time}")
    p = prediction["accuracy"]
    print("Prediction accuracy:\n", p)

    logging.info(f"Prediction accuracy:{p}")

if __name__ == "__main__":
    main()
