CONFIG = {
"DELTA_X_LEAK" : 1,
"NO_LEAK_PORTION" : 0.1,
"EPOCH" : 50,
"PARALLEL" : True,
"GEOJSON" : "./input/geojson.json",
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
}

}