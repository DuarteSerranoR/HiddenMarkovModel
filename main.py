from hmm.hmm import HMM
from lib.logger import logger as log

import pandas as pd
import numpy as np


def init_data(obs_path,res_path):
    
    if not obs_path:
        raise Exception("Observations path not set!")
    if not res_path:
        raise Exception("Trainning results path not set!")

    


    # TODO - load train data
    raise NotImplemented("Load train data - observations + original_result")
    #

    log.info("Data loaded to memory.")
    return obs, og_res


def init_data(train_path):
    
    if not train_path:
        raise Exception("Train path not set!")

    


    # TODO - load train data
    raise NotImplemented("Load train data - observations + original_result")
    #

    log.info("Data loaded to memory.")
    return data_df



# Depedencies to install:
#   - numpy

if __name__ == "__main__":
    log.info("Started HMM model testing")
    train_obs,train_og_res = init_data("","") # TODO - input paths
    #train_df = init_data("") # TODO - input paths using dataframes
    
    
    model = HMM()
    tr#ain_out, train_accuracy = model.train_supervised_numpy(train_df, smoothing = 0.3, test = True)
    train_out, train_accuracy = model.train_supervised_numpy(train_obs, train_og_res, smoothing = 0.3, test = True)
    
    # TODO - load test data for f1
    raise NotImplemented("Load test data - observations + wanted_result")
    #

    output = model.compute(input)
    # TODO - f1 not implemented yet, use wanted_result var
    print(output)
