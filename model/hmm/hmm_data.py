import pickle

from logger import logger as log

import numpy as np

class HMM_ModelData:
    
    train_method: str
    loaded: bool = False
    
    num_states: int
    
    initial_probabilities: np.ndarray
    transition_probabilities: np.ndarray
    final_probabilities: np.ndarray
    emission_probabilities: np.ndarray
    
    @classmethod
    def from_memory(cls, train_method: str, initial_probabilities: np.ndarray, transition_probabilities: np.ndarray, final_probabilities: np.ndarray, emission_probabilities: np.ndarray):
        model: HMM_ModelData = HMM_ModelData()
        
        model.train_method = train_method
        model.initial_probabilities = initial_probabilities
        model.transition_probabilities = transition_probabilities
        model.final_probabilities = final_probabilities
        model.emission_probabilities = emission_probabilities
        
        model.num_states = len(initial_probabilities)
        
        model.loaded = True
        
        log.info("Model HMM saved to memory.")
        
        return model

    @classmethod
    def from_disk(cls, path: str):
        model: HMM_ModelData = pickle.load(open(path, 'rb'))
        if model is None:
            raise Exception("ValidationException: Could not load this model file!!")
        self = model
        
        self.loaded = True
        
        log.info("Model HMM loaded from disk.")
        
        return model

    def to_disk(self, path: str):
        if not self.loaded:
            raise Exception("ValidationException: Model not loaded!")

        pickle.dump(self, open(path, 'wb'))
        
        log.info("Model HMM saved to disk at '{0}'.".format(path))
        
