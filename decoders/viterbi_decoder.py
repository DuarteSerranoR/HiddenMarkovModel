
import numpy as np


class ViterbiDecoder:

    @classmethod
    def viterbi_decode(self, initial_scores, transition_scores, final_scores, emission_scores):
        
        length = np.size(emission_scores, 0)
        num_states = np.size(initial_scores)
        
        viterbi = np.zeros([length, num_states]) + -np.inf # viterbi scores
        backtrack = -np.ones([length, num_states], dtype=int) # viterbi path
        
        best_path = -np.ones(length, dtype=int)
        
        # Compute scores
        for k in range(num_states):
            viterbi[0,k] = initial_scores[k] + emission_scores[np.argmax(0),k]

        for i in range(1,length):
            for k in range(num_states):
                viterbi[i,k] = np.amax(transition_scores[i-1, k, :] + viterbi[i-1, :]) + emission_scores[i, k]
                backtrack[i,k] = np.argmax(transition_scores[i-1, k, :] + viterbi[i-1, :])
        
        #best_score = np.amax(final_scores + viterbi[num_states - 1, :])
        
        
        best_path[length - 1] = np.argmax(final_scores + viterbi[length - 1, :])
        for i in reversed(range(0, length - 1)):
            best_path[i] = backtrack[i + 1, best_path[i + 1]]
        
        # Compute likelihood
        
            
        #if ( \
        #    np.isinf(self.matrix/array).any()
        #):
        #    raise ValueError("Found infinite value convert values to the log and change operations of division \
        #                     to subtraction and multiplication to sum!!")
        
        return best_path#, best_score
    