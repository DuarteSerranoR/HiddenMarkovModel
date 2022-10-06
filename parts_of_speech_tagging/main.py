from hmm.hmm import HMM
from lib.logger import logger as log

import pandas as pd
import numpy as np

def init_data(train_path): #obs_path,res_path):
    
    #if not obs_path:
    #    raise Exception("Observations path not set!")
    #if not res_path:
    #    raise Exception("Trainning results path not set!")

    # ---
    # used dataset -> https://www.cnts.ua.ac.be/conll2000/chunking/

    data_file = open(train_path,"r")
    data_lines = [ line.replace("\n","") for line in data_file.readlines()]# if line != ", , O\n" ]
    data_file.close()
    data_raw = [ [line.split(" ")[0],line.split(" ")[2]] if line != "" else ("","") for line in data_lines ]
    data_np = np.array(data_raw,dtype=np.object0)
    data_words,data_state = data_np[:,0],data_np[:,1]
    data_df = pd.DataFrame({ "words": data_words, "state": data_state })
    #data_df.to_csv("./temp.csv")

    # ---

    log.info("Data loaded to memory.")
    return data_df #obs, og_res



# Depedencies to install:
#   - numpy

if __name__ == "__main__":
    log.info("Started HMM model testing")
    #train_obs,train_og_res = init_data("./parts_of_speech_tagging/data/train.txt")
    train_df = init_data("./parts_of_speech_tagging/data/train.txt")
    
    
    model = HMM()
    train_out, train_accuracy = model.train_supervised_numpy(train_df, smoothing = 0.3, test = True)
    
    # TODO - load test data for f1
    raise NotImplemented("Load test data - observations + wanted_result")
    #

    output = model.compute(input)
    # TODO - f1 not implemented yet, use wanted_result var
    print(output)
