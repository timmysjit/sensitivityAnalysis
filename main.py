
import json
from generator import *
from generator_parallel import generate_leakage_dataset_parallel
from model_training import *
import time

def main():
    start_time = time.perf_counter()
    file = open("geojson.json", "r")
    data = json.load(file)
    file.close()

    hexagon_geojson = json.loads(data["hexagonGeoJson"])
    delta_x_leak = 1
    demand_range = [0, 10, 0.05]
    no_leak_portion = 0.1
    noise_amplitude = 0.05
    network_inp_content = data["inpContent"]
    epoch_count = 100
    crs = data["crsCode"]
    hexagon_radius = data["hexagonRadius"]
    step3_sensor_attachments = data["step3SelectedNodeIds"]

    output_file = "train_data.jsonl"
    # gen_result = generate_leakage_dataset_parallel(
    #     hexagon_geojson=hexagon_geojson,
    #     network_inp_content=network_inp_content,
    #     crs=crs,
    #     delta_x_leak=delta_x_leak,
    #     hexagon_radius=hexagon_radius,
    #     step3_sensor_attachments=step3_sensor_attachments,
    #     leak_demand_range=demand_range,
    #     no_leak_portion=no_leak_portion,
    #     noise_amplitude=noise_amplitude,
    #     num_workers=16,
    #     output_file=output_file,
    # )

    dataset = []
    with open(output_file) as f:
        for line in f:
            dataset.append(json.loads(line))
    print(f"Loaded {len(dataset)} samples from {output_file}")

    dataset_payload = {"dataset": dataset}

    train_output = train_leakage_model(dataset_payload=dataset_payload, network_inp_content=network_inp_content, epoch_count=epoch_count)
    gcn_model_id = train_output["modelId"]
    tl_output = transfer_learning(model_id=gcn_model_id, dataset_payload=dataset_payload, network_inp_content=network_inp_content, epoch_count=epoch_count)

    test_dataset = generate_random_test_data(
        hexagon_geojson=hexagon_geojson,
        network_inp_content=network_inp_content,
        crs=crs,
        delta_x_leak=delta_x_leak,
        hexagon_radius=hexagon_radius,
        step3_sensor_attachments=step3_sensor_attachments,
        num_samples=1000,
        leak_demand_min=demand_range[0],
        leak_demand_max=demand_range[1],
        noise_amplitude=noise_amplitude,
        no_leak_portion=no_leak_portion,
    )

    tl_model_id = tl_output["modelId"]
    prediction = predict_leakage(model_id=tl_model_id, test_dataset=test_dataset["dataset"], network_inp_content=network_inp_content)

    end_time = time.perf_counter()

    print("Time taken: ", end_time - start_time)
    print("Prediction accuracy:\n", prediction["accuracy"])

if __name__ == "__main__":
    main()
