import os
from typing import List
from xml.dom import InvalidAccessErr
import numpy as np

from lib.logger import logger as log

from hmm.hmm_data import HMM_ModelData
from hmm.train.hmm_numpy_train import HMM_NumpyTrain

from decoders.decoder import HMM_Decoder


class HMM:
    
    trained: bool
    model_path: str
    decoder: HMM_Decoder

    def __init__(self, model_path="./data/hmm_model.dat"):
        self.model_path = model_path
        if os.path.exists(model_path):
            model_data: HMM_ModelData = HMM_ModelData.from_disk(model_path)
            self.decoder = HMM_Decoder(model_data)
            self.trained = True
        else:
            self.trained = False

    
    # Train methods

    def train_supervised_numpy(self, x_in: str, y_in: str, smoothing = 0, test = False) -> str:
        """
            x_in: observation
            y_in: the result we want to obtain when applied the states into the observation
        """
        
        if not self.trained or test:
            if not self.trained:
                if os.path.exists(self.model_path):
                    raise FileExistsError("File found on referenced path to train and save the model.")

                # Generate Initial/Transition/Estimated Probabilities
                
                # Train
                log.debug("Training the model...")
                x,y = HMM_NumpyTrain.pre_process(x_in, y_in)
                train_set = HMM_NumpyTrain(x, y, smoothing)
                train_set.compute_counts()
                train_set.compute_probabilities()
                model_data: HMM_ModelData = train_set.get_trained_model_data()
                log.debug("Trained!")
                
                train_set = None
                
                model_data.to_disk(self.model_path)
                
                self.decoder = HMM_Decoder(model_data)
                
                self.trained = True
            
            
            else:
                x = self.__pre_process_data(x_in)
                log.warning("TRAIN TEST WITH MODEL LOADED FROM DISK!!")


            # Decode
            #decoded_prediction, total_score = self.decoder.viterbi_decode(x)
            decoded_prediction = self.decoder.viterbi_decode(x)
            
            out_str = self.__compute_output(input, decoded_prediction)

            # Compute accuracy
            accurate_count = 0
            accuracy_total = 0
            
            # TODO - compute accuracy
            raise NotImplemented("Accuracy not Implemented!")
            #
            
            accuracy = accurate_count / accuracy_total 
            # TODO - measure f1 scores and use test files to check the accuracy for them too

            log.info("model trained with accuracy of {0}%".format(accuracy*100))
            return out_str, accuracy
            
        else:
            raise Exception("ValidationException: Model already trained!")

    #def train_supervised_torch(self, x, y):
    #    if not self.trained:
    #        if os.path.exists(self.model_path):
    #            raise FileExistsError("File found on referenced path to train and save the model.")
    #        
    #        # For each line prepare each character pattern
    #
    #        # Generate Initial/Transition/Estimated Probabilities
    #        
    #        
    #        raise NotImplementedError("Torch training not implemented yet!")
    #        # if gpu
    #        #   train_method = "Torch-GPU"
    #        # else
    #        train_method = "Torch-CPU"
    #        self.trained = True
    #    
    #    else:
    #        raise ValidationException("Model already trained!")
    
    
    
    # Usage methods
        
    def compute(self, input: str) -> str:
        if self.trained:
            input_coded = self.__pre_process_data(input)
            decoded_prediction = self.decoder.viterbi_decode(input_coded)
            out_str = HMM.__compute_output(input, decoded_prediction)
            return out_str
        else:
            raise InvalidAccessErr("Model not trained or loaded for use!!")
    
    
    #def process_batch(self, input: List[str]):
    #    input = self.viterbi_decoder.__pre_process_batch(input)
    #    (initial_scores, transition_scores, final_scores, emission_scores) = self.viterbi_decoder.compute_scores(input)
    #    y = self.viterbi_decoder.viterbi_decode(input, initial_scores, transition_scores, final_scores, emission_scores)
    #    output = self.__compute_output(y)
    #    return output
    #




    
    # Private methods


    def __pre_process_data(self, input: str) -> np.ndarray:
        
        return self.__compute_pre_process(input)
        
    #def __pre_process_batch(self, input: np.ndarray) -> np.ndarray:
    #    
    #    input = input.
    #    
    #    return self.__compute_pre_process(y_str)
    
    def __compute_pre_process(self, input_str: np.ndarray) -> np.ndarray:
        length = len(input_str)
        
        if input_str[length - 1] == "":
            input_str = input_str[0:length - 2]
        
        # observation sequence
        sequence: List[List[int]] = []

        
        char_total_prios = dict()
        # priors of 2 chars posterior and anterior
        # priors of 3 chars
        char_seq_dict_i = 0

        used_line_ix = -1
        
        for line_i in range(len(input_str)): # index of each line
            if len(input_str[line_i]) < 2:
                continue
            
            sequence.append([])
            used_line_ix += 1
            
            line_space_i = 0
            spaces = input_str[line_i].split(" ")
            
            for char_i in range(0, len(input_str[line_i])): # index of each character
                
                char = input_str[line_i][char_i]
                
                if char == " ":
                    if line_space_i + 1 < len(spaces):
                        word = spaces[line_space_i] + " " + spaces[line_space_i + 1]
                        line_space_i += 1
                    else:
                        word = spaces[line_space_i] + " "
                else:
                    word = spaces[line_space_i]
                
                if char_i == 0:
                    #char_seq2_ant = x_str[line_i][char_i]
                    char_seq3 = input_str[line_i][char_i] + input_str[line_i][char_i + 1]
                    #char_seq2_post = x_str[line_i][char_i] + x_str[line_i][char_i + 1]
                elif char_i + 1 == len(input_str[line_i]):
                    #char_seq2_ant = x_str[line_i][char_i - 1] + x_str[line_i][char_i]
                    char_seq3 = input_str[line_i][char_i - 1] + input_str[line_i][char_i]
                    #char_seq2_post = x_str[line_i][char_i]
                else:
                    #char_seq2_ant = x_str[line_i][char_i - 1] + x_str[line_i][char_i]
                    char_seq3 = input_str[line_i][char_i - 1] + input_str[line_i][char_i] + input_str[line_i][char_i + 1]
                    #char_seq2_post = x_str[line_i][char_i] + x_str[line_i][char_i + 1]
                    
                #char_total_seq = (char_seq2_ant, char_seq3, char_seq2_post)
                char_total_seq = (char_seq3, word)
                    
                if not char_total_seq in char_total_prios:
                    char_total_prios[char_total_seq] = char_seq_dict_i
                    char_seq_dict_i += 1
                
                sequence[used_line_ix].append(char_total_prios[char_total_seq])

        sequence = np.array(sequence, dtype="object")

        return sequence

    # TODO - repeated in HMMNumpyTrain class, put in encoder as class method
    @classmethod
    def __compute_output(self, input: list(), decoded_prediction: np.ndarray) -> str:
        """
        input -> is the input
        decoded_prediction -> is the decoded states to apply into the input

        # output -> is the output of the state applied into the input
        """
        
        # TODO - compute the state aplications into the input
        raise "Not Implemented!"
        
        return output

