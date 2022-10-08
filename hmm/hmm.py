import os
import numpy as np
import pandas as pd
from plistlib import InvalidFileException

from lib.logger import logger as log

from hmm.hmm_data import HMM_ModelData
from hmm.train.hmm_numpy_train import HMM_NumpyTrain

from decoders.decoder import HMM_Decoder


class HMM:
    
    trained: bool
    model_path: str
    decoder: HMM_Decoder

    def __init__(self, model_path="./model/hmm.dat"):
        self.model_path = model_path
        if os.path.exists(model_path):
            model_data: HMM_ModelData = HMM_ModelData.from_disk(model_path)
            self.decoder = HMM_Decoder(model_data)
            self.trained = True
        else:
            self.trained = False

    
    # Train methods

    # TODO - implement different types of WER, WAcc, or f score tests out of the box !!
    def train_numpy(self, x_in = False, y_in = False, df_in: pd.DataFrame = False, x_test = False, y_test = False, test_df: pd.DataFrame = False, smoothing = 0, test = False):
        """
            df_in: DataFrame with observations and states
            
            or

            x_in: observation
            y_in: the result we want to obtain when applied the states into the observation
        """

        if not isinstance(df_in, pd.DataFrame) and (not isinstance(x_in, str) or not isinstance(y_in, str)):
            raise ValueError("Invalid Arguments, you need to supply either train dataframe or input observation and input objective")

        if self.trained and not test:
            raise ValueError("Test already trained and test bool set to false, so there is nothing to do!")
        
        if not self.trained:
            if os.path.exists(self.model_path):
                raise FileExistsError("File found on referenced path to train and save the model.")

            # Generate Initial/Transition/Estimated Probabilities
            
            # Train
            log.debug("Training the model...")
            if isinstance(df_in, pd.DataFrame):
                x,y,tokenizer = HMM_NumpyTrain.pre_process(df_in)
            else:
                x,y,tokenizer = HMM_NumpyTrain.pre_process(x_in, y_in)

            train_set = HMM_NumpyTrain(x, y, smoothing, tokenizer)
            train_set.compute_counts()
            train_set.compute_probabilities()
            model_data: HMM_ModelData = train_set.get_trained_model_data()
                
            log.debug("Trained!")
            
            train_set = None
            
            model_data.to_disk(self.model_path)
            
            self.decoder = HMM_Decoder(model_data)
            
            self.trained = True
            
            
        else:
            log.warning("TRAIN TEST WITH MODEL LOADED FROM DISK!!")
        
        if test:

            if not isinstance(df_in, pd.DataFrame) and (not isinstance(x_in, str) or not isinstance(y_in, str)):
                raise ValueError("Invalid Arguments, you need to supply either train dataframe or input observation and input objective for evaluation, if the Test bool is active.")
                
            return self.eval(x_test, y_test, test_df)
            
            
        #else:
        #    raise Exception("ValidationException: Model already trained!")

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
        
    def compute(self, input):
        if self.trained:
            input_coded = self.__pre_process_observation(input)
            decoded_prediction = self.decoder.viterbi_decode(input_coded)
            if self.decoder.model_data.with_tokenizer:
                out = HMM.__compute_output(input, decoded_prediction, self.decoder.model_data.tokenizer)
            else:
                out = HMM.__compute_output(input, decoded_prediction)
            return out
        else:
            raise InvalidFileException("Model not trained or loaded for use!!")

    # TODO - self_supervised -> suggestion

    def eval(self, x_test = False, y_test = False, test_df: pd.DataFrame = False):
        
        # TODO - evaluate, print or return the desired outcome

        raise NotImplemented("Model evaluation not Implemented!")
        
    
    #def process_batch(self, input: List[str]):
    #    input = self.viterbi_decoder.__pre_process_batch(input)
    #    (initial_scores, transition_scores, final_scores, emission_scores) = self.viterbi_decoder.compute_scores(input)
    #    y = self.viterbi_decoder.viterbi_decode(input, initial_scores, transition_scores, final_scores, emission_scores)
    #    output = self.__compute_output(y)
    #    return output
    #




    
    # Private methods

    def __pre_process_observation(self, input: str) -> np.ndarray:
        
        sequence = [[]] # computed in batches / lines

        # TODO - compute pre-processing of input string into matrix
        raise NotImplemented("You need to preprocess input into suitable format.")

        if self.decoder.model_data.with_tokenizer:
            raise NotImplemented()

        sequence = np.array(sequence, dtype="object")

        return sequence

    @classmethod
    def __compute_output(self, input: list(), decoded_prediction: np.ndarray, tokenizer = False) -> str:
        """
        input -> is the input
        decoded_prediction -> is the decoded states to apply into the input

        --
        
        output -> is the output of the state applied into the input
        """
        
        # TODO - compute and return the predicted states
        raise NotImplemented("Not Implemented!")

        if tokenizer != False:
            raise NotImplemented("Use tokenizer to resolve tokens from the prediction.")
        
        return output
