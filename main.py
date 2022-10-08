from hmm.hmm import HMM
from lib.logger import logger as log

import pandas as pd


def init_data(train_path, obj_path = ""): # NOTE - train_path or obs_path if you want to use two files to train instead of just one
    
    if not train_path:
        raise Exception("Train/Observation path not set!")


    # TODO - load train data
    raise NotImplemented("Load train data - observations + original_result")
    #

    log.info("Data loaded to memory.")
    #return data_df
    #return obs, obj


# Depedencies to install:
#   - numpy

if __name__ == "__main__":
    log.info("Started HMM model testing")
    train_data = init_data("") # TODO - input paths
    
    model = HMM()
    if isinstance(train_data, pd.DataFrame):
        train_out, train_accuracy = model.train_numpy(df_in=train_data, smoothing = 0.3, test = True)
    else:
        train_obs, train_obj = train_data
        train_out, train_accuracy = model.train_numpy(x_in=train_obs, y_in=train_obj, smoothing = 0.3, test = True)

    #output = model.compute(input)
    #print(output)
