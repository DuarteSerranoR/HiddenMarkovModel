from typing import List
from itertools import chain


from hmm.hmm_data import HMM_ModelData
from logger import logger as log

import numpy as np


class HMM_NumpyTrain:
    
    # All numpy lists
    
    x: np.ndarray
    y: np.ndarray
    
    initial_counts: np.ndarray
    transition_counts: np.ndarray
    final_counts: np.ndarray
    emission_counts: np.ndarray
    
    initial_probabilities: np.ndarray
    transition_probabilities: np.ndarray
    final_probabilities: np.ndarray
    emission_probabilities: np.ndarray
    
    
    lines_len: int
    distinct_labels_len: int
    distinct_states_len: int
    
    def __init__(self, x, y, smoothing=0):
        self.x = x
        self.y = y
        _y = []
        [ _y.append(l) for l in y ]
        
        self.distinct_labels_len = max(map(max, x)) + 1
        self.distinct_states_len = len(np.unique(list(chain.from_iterable(y))))
        
        
        dim = self.distinct_states_len
        
        self.initial_counts = np.zeros(dim, dtype=np.float32)
        self.initial_probabilities = np.zeros(dim, dtype=float)
        
        self.final_counts = np.zeros(dim, dtype=np.float32)
        self.final_probabilities = np.zeros(dim, dtype=float)
        
        
        dim = (self.distinct_states_len,self.distinct_states_len)
        
        self.transition_counts = np.zeros(dim, dtype=np.float32)
        self.transition_probabilities = np.zeros(dim, dtype=float)
        
        
        dim = (self.distinct_labels_len, self.distinct_states_len)
        
        self.emission_counts = np.zeros(dim, dtype=np.float32)
        self.emission_probabilities = np.zeros(dim, dtype=float)
        
        self.smoothing = smoothing
    
    @staticmethod
    def pre_process(x_in: str, y_in: str):
        """
        This is suposed to take in an observation input, and what we want to make out of it, 
        and return the states applied to said input and the out format of the input for each state.

        x_in -> inputed value
        y_in -> the trainning data to where you want to convert the inputed value

        x -> the input sequence / observations
        y -> the states of said input sequence / states
        """
        
        log.info("Pre-processing training data...")

        if len(x_in) != len(y_in): # number of lines
            raise OverflowError("Lengths of input training data are not equal!")

        # observation sequence
        x: List[List[int]] = []

        # state sequence 
        y: List[List[int]] = []

        # TODO - appply your trainning/state_processing logic

        raise "Pre-Processing Not Implemented!"

        #

        x = np.array(x, dtype="object")
        y = np.array(y, dtype="object")

        # Test the decoded prediction of said states, needs to be equal to y_in
        train_test = HMM.__compute_output(x_in,y)
        train_test = np.array(train_test.splitlines(), dtype=str)
        
        if (train_test != y_in).all():
            raise Exception("Data from train doesn't match input!! Training bug or bad input.")

        log.info("Training data ready to use!")
        return x, y

    def compute_counts(self):
        x = self.x
        y = self.y
        
        for ix in range(len(y)):
        
            ### Initial Counts && Final Counts ###
            
            ## y - get the uniqueness of each sequence (line) in index=0
            _y = y[ix][0]
            self.initial_counts[_y] += 1
            
            ## y - get the uniqueness of each sequence (line) in index=(length - 1)
            _y = y[ix][len(y[ix]) - 1]
            self.final_counts[_y] += 1
            
            ### Transition Counts && Emission Counts ###
            for i in range(len(y[ix])):
                _x = x[ix][i]
                _y = y[ix][i]
            
                # Emission
                self.emission_counts[_x,_y] += 1
                
                # Transition
                if i != 0:
                    _y_prev = y[ix][i-1]
                    self.transition_counts[_y,_y_prev] += 1
        
        log.info("Computed Counts")
    
    def compute_probabilities(self):
        
        ### Initial Probabilities ###
        
        if self.smoothing != 0:
            np.add(self.initial_counts, self.smoothing, out=self.initial_counts)
            np.add(self.transition_counts, self.smoothing, out=self.transition_counts)
            np.add(self.final_counts, self.smoothing, out=self.final_counts)
            np.add(self.emission_counts, self.smoothing, out=self.emission_counts)
        
        # initial counts / sum(initial counts)
        self.initial_probabilities = self.initial_counts / np.sum(self.initial_counts)
        
        ### Transition Probabilities ###
        
        # transition counts / ( sum(transition counts on first axis) + final counts )
        self.transition_probabilities = self.transition_counts / (np.sum(self.transition_counts, 0) + self.final_counts)
        
        ### Final Probabilities ###
        
        # final counts / ( sum(transition counts on first axis) + final counts )
        self.final_probabilities = self.final_counts / (np.sum(self.transition_counts, 0) + self.final_counts)
        
        ### Emission Probabilities ###
        
        # final emission / ( sum(emission counts on first axis) )
        self.emission_probabilities = self.emission_counts / np.sum(self.emission_counts, 0)
        
        
        
        log.info("Computed Probabilities")
    
    def get_trained_model_data(self) -> HMM_ModelData:
        model_data = HMM_ModelData.from_memory("Numpy", self.initial_probabilities, self.transition_probabilities, self.final_probabilities, self.emission_probabilities)
        return model_data

