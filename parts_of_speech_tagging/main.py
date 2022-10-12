from hmm.hmm import HMM
from lib.logger import logger as log

import pandas as pd
import numpy as np
import lib.download_nltk


def init_data(train_path, obj_path = ""): # NOTE - train_path or obs_path if you want to use two files to train instead of just one
    
    if not train_path:
        raise Exception("Train/Observation path not set!")

    # ---
    # used dataset -> https://www.cnts.ua.ac.be/conll2000/chunking/

    data_file = open(train_path,"r")
    data_lines = [ line.replace("\n","") for line in data_file.readlines()]
    data_file.close()
    data_raw = [ [line.split(" ")[0],line.split(" ")[2]] if line != "" else ("","") for line in data_lines ]
    for line_ix in range(len(data_raw)):
        if "-" in data_raw[line_ix][1]:
            data_raw[line_ix][1] = data_raw[line_ix][1].split("-")[1]
    #    if data_raw[line_ix][0] == "Corp.":
    #        data_raw[line_ix][0] = "Corp"
    #    if data_raw[line_ix][0] == "Mr.":
    #        data_raw[line_ix][0] = "Mr"
    data_np = np.array(data_raw,dtype=np.object0)
    data_words,data_state = data_np[:,0],data_np[:,1]
    data_df = pd.DataFrame({ "words": data_words, "states": data_state })
    #data_df.to_csv("./temp.csv")

    # ---

    log.info("Data loaded to memory.")
    return data_df
    #return obs, obj



# Depedencies to install:
#   - numpy

if __name__ == "__main__":
    log.info("Started HMM model testing")
    train_data = init_data("./parts_of_speech_tagging/data/train.txt")
    test_df = init_data("./parts_of_speech_tagging/data/test.txt")
    
    model = HMM(model_path="./parts_of_speech_tagging/model/hmm.dat")

    #model.eval(test_df=test_df)

    #if isinstance(train_data, pd.DataFrame):
    #    #model.train_numpy(df_in=train_data, smoothing = 0.3)
    #    model.train_numpy(df_in=train_data, test_df=test_df, smoothing = 0.3, test = True)
    #else:
    #    train_obs, train_obj = train_data
    #    train_out, train_accuracy = model.train_numpy(x_in=train_obs, y_in=train_obj, smoothing = 0.3, test = True)

    output = model.compute("Can this be possible? I do not think so, this does not seem like a good idea.")
    print(output)
 