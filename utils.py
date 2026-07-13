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