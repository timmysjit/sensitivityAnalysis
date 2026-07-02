import os
import json
from config import *
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

def save(train_data , train, tl_data, tl_train, test_data, prediction, geojson, id : int):
    file = open(f"./data/run{id}/config.json", "w")
    json.dump(CONFIG, file, indent = 4)
    file.close()

    file = open(f"./data/run{id}/geojson.json", "w")
    json.dump(geojson, file, indent = 4)
    file.close()

    if (CONFIG["TRAIN"]["save"]):
        file = open(f"./data/run{id}/train_data.json", "w")
        json.dump(train_data, file, indent = 4)
        file.close()
        file = open(f"./data/run{id}/train.json", "w")
        json.dump(train, file, indent = 4)
        file.close()
    if (CONFIG["TL"]["save"]):
        file = open(f"./data/run{id}/tl_data.json", "w")
        json.dump(tl_data, file, indent = 4)
        file.close()
        file = open(f"./data/run{id}/tl_train.json", "w")
        json.dump(tl_train, file, indent = 4)
        file.close()
    if (CONFIG["TEST"]["save"]):
        file = open(f"./data/run{id}/test_data.json", "w")
        json.dump(test_data, file, indent = 4)
        file.close()
    file = open(f"./data/run{id}/prediction.json", "w")
    json.dump(prediction, file, indent = 4)
    file.close()