CONFIG = {
"DELTA_X_LEAK" :2,
"NO_LEAK_PORTION" : 0.1,
"EPOCH" : 50,
"TRAIN" : {
    "demand_range" : [0, 5, 0.05],
    "noise_amplitude" : 0.05,
    "save" : False
},


"TL" : {
    "num_samples" : 1,
    "leak_demand_min" : 0,
    "leak_demand_max" : 5,
    "noise_amplitude" : 0.05,
    "save" : False
},

"TEST" : {
    "num_samples" : 1,
    "leak_demand_min" : 0,
    "leak_demand_max" : 5,
    "noise_amplitude" : 0.05,
    "save" : False
}

}