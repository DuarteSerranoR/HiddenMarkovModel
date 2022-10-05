
from model.hmm.hmm_data import HMM_ModelData
from decoders.viterbi_decoder import ViterbiDecoder


import numpy as np


class HMM_Decoder:
    
    #viterbi: ViterbiDecoder
    #minimum_risk: MinimunRiskDecoder
    
    model_data: HMM_ModelData
    
    def __init__(self, model_data):
        self.viterbi = ViterbiDecoder()
        self.model_data = model_data
    
    def viterbi_decode(self, input):
        
        prediction = []
        #total_score = 0
        for line in input:
            (initial_scores, transition_scores, final_scores, emission_scores) = self.__compute_scores(line)
            #_prediction, _total_score = ViterbiDecoder.viterbi_decode(initial_scores, transition_scores, final_scores, emission_scores)
            _prediction = ViterbiDecoder.viterbi_decode(initial_scores, transition_scores, final_scores, emission_scores)
            
            prediction.append(_prediction)
            #total_score += _total_score

        #total_score = total_score / len(input)

        
        return prediction#, total_score

    def __compute_scores(self, input):
        
        length = len(input)
        num_states = self.model_data.num_states
        
        ### Initial Scores && Final Scores ###
        
        # Initial Scores
        initial_scores = np.log(self.model_data.initial_probabilities)
        
        # Final Scores
        final_scores = np.log(self.model_data.final_probabilities)
        
        
        
        ### Transition Scores && Emission Scores ###
        
        emission_scores = np.zeros((length, num_states)) + -np.inf
        transition_scores = np.zeros((length - 1, num_states, num_states)) + -np.inf
        
        for i in range(length):
            # Emission Scores
            emission_scores[i, :] = np.log(self.model_data.emission_probabilities[input[i], :])
            
            # Transition Scores
            if i > 0:
                transition_scores[i - 1, :, :] = np.log(self.model_data.transition_probabilities)
        
        
        return (initial_scores, transition_scores, final_scores, emission_scores)
        
    