import json
from generator import *
from model_training import *
import time
from config import *
from utils import *
import logging

def main():
    id = save_init()
    start_time = time.perf_counter()
    file = open("geojson.json", "r")
    data = json.load(file)
    file.close()

    hexagon_geojson = json.loads(data["hexagonGeoJson"])
    network_inp_content = data["inpContent"]
    crs = data["crsCode"]
    hexagon_radius = data["hexagonRadius"]
    step3_sensor_attachments = data["step3SelectedNodeIds"]

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

    print("training started")
    train_output = train_leakage_model(dataset_payload=train_data, network_inp_content=network_inp_content, epoch_count=CONFIG["EPOCH"])
    print("training finished")
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

    print("transfer learning training started")
    tl_output = transfer_learning(model_id=gcn_model_id, dataset_payload=tl_data, network_inp_content=network_inp_content, epoch_count=CONFIG["EPOCH"])
    print("transfer learning training finished")
    tl_model_id = tl_output["modelId"]

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
    
    print("prediction on test data started")
    prediction = predict_leakage(model_id=tl_model_id, test_dataset=test_data["dataset"], network_inp_content=network_inp_content)
    print("prediction on test data finished")

    end_time = time.perf_counter()

    print("Time taken: ", end_time - start_time)
    logging.info(f"Time taken: {end_time - start_time}")
    p = prediction["accuracy"]
    print("Prediction accuracy:\n", p)

    logging.info(f"Prediction accuracy:{p}")

    print("saving files")
    save(train_data, train_output, tl_data, tl_output, test_data, prediction, hexagon_geojson, id)
    print("finished saving")

if __name__ == "__main__":
    main()
