from hmm.hmm import HMM
from logger import logger as log


def init_data(obs_path,res_path):
    
    # TODO - load train data
    raise NotImplemented("Load train data - observations + original_result")
    #

    log.info("Data loaded to memory.")
    return obs, og_res



# Depedencies to install:
#   - numpy

if __name__ == "__main__":
    log.info("Started HMM model testing")
    train_obs,train_og_res = init_data("","") # TODO - input paths
    
    
    model = HMM()
    train_out, train_accuracy = model.train_supervised_numpy(train_obs, train_og_res, smoothing = 0.3, test = True)
    
    # TODO - load test data for f1
    raise NotImplemented("Load test data - observations + wanted_result")
    #

    output = model.compute(input)
    # TODO - f1 not implemented yet, use wanted_result var
    print(output)
