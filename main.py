import json
from generator import *
from model_training import *
import time

def main():
    start_time = time.perf_counter()
    file = open("geojson.json", "r")
    data = json.load(file)
    file.close()

    hexagon_geojson = json.loads(data["hexagonGeoJson"])
    delta_x_leak = 20
    demand_range = [2, 2, 0.05]
    no_leak_portion = 0
    noise_amplitude = 0.05
    network_inp_content = data["inpContent"]
    
    crs = data["crsCode"]
    hexagon_radius = data["hexagonRadius"]
    step3_sensor_attachments = data["step3SelectedNodeIds"]

    train_data = generate_leakage_dataset(
    hexagon_geojson = hexagon_geojson,
    network_inp_content = network_inp_content,
    crs = crs,
    delta_x_leak = delta_x_leak,
    hexagon_radius = hexagon_radius,
    step3_sensor_attachments = step3_sensor_attachments,
    leak_demand_range = demand_range,
    no_leak_portion = no_leak_portion,
    noise_amplitude = noise_amplitude,
    ) 

    epoch_count = 1
    train_output = train_leakage_model(dataset_payload=train_data, network_inp_content=network_inp_content, epoch_count=epoch_count)

    gcn_model_id = train_output["modelId"]


    tl_num_samples = 1
    tl_leak_demand_min = 2
    tl_leak_demand_max = 2
    tl_noise_amplitude = 0.05

    tl_data = generate_random_test_data(
    hexagon_geojson = hexagon_geojson,
    network_inp_content = network_inp_content,
    crs = crs,
    delta_x_leak = delta_x_leak,
    hexagon_radius = hexagon_radius,
    step3_sensor_attachments = step3_sensor_attachments,
    num_samples = tl_num_samples,
    leak_demand_min = tl_leak_demand_min,
    leak_demand_max = tl_leak_demand_max,
    noise_amplitude = tl_noise_amplitude,
    no_leak_portion = no_leak_portion,
    )

    tl_output = transfer_learning(model_id=gcn_model_id, dataset_payload=tl_data, network_inp_content=network_inp_content, epoch_count=epoch_count)
    tl_model_id = tl_output["modelId"]

    test_num_samples = 1
    test_leak_demand_min = 2
    test_leak_demand_max = 2
    test_noise_amplitude = 0.05

    test_dataset = generate_random_test_data(
        hexagon_geojson=hexagon_geojson,
        network_inp_content=network_inp_content,
        crs=crs,
        delta_x_leak=delta_x_leak,
        hexagon_radius=hexagon_radius,
        step3_sensor_attachments=step3_sensor_attachments,
        num_samples=test_num_samples,
        leak_demand_min=test_leak_demand_min,
        leak_demand_max=test_leak_demand_max,
        noise_amplitude=test_noise_amplitude,
        no_leak_portion=no_leak_portion,
    )

    prediction = predict_leakage(model_id=tl_model_id, test_dataset=test_dataset["dataset"], network_inp_content=network_inp_content)

    end_time = time.perf_counter()

    print("Time taken: ", end_time - start_time)
    print("Prediction accuracy:\n", prediction["accuracy"])

if __name__ == "__main__":
    main()
