CONFIG = {
"DELTA_X_LEAK" : 2.5,
"NO_LEAK_PORTION" : 0.1,
"EPOCH" : 50,
"PARALLEL" : True,
"PASS_DATA_GENERATION" : True,
"DATA_GENERATION_DIR" : "./leaks",
"GEOJSON" : "./input/geojson4_200_0_0.json",
"CHANGE_GEOJSON" : None,
"TRAIN" : {
    "demand_range" : [0, 2, 0.01],
    "noise_amplitude" : 0.05,
    "save" : False
},


"TL" : {
    "num_samples" : 1000,
    "leak_demand_min" : 0,
    "leak_demand_max" : 2,
    "noise_amplitude" : 0.1,
    "save" : False
},

"TEST" : {
    "num_samples" : 1000,
    "leak_demand_min" : 0,
    "leak_demand_max" : 2,
    "noise_amplitude" : 0.1,
    "save" : False
},

"SENSITIVITY" : {
    "sensor_counts" : list(range(1, 11)),
    "repeats" : 3,
    "base_seed" : 7,
    "output_root" : "./data/sensor_sensitivity",
    "continue_on_error" : True,
    "save_shared_datasets" : False,
    "save_prediction_details" : True,
    "save_model_outputs" : True
}

}