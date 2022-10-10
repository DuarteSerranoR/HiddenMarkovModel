import os
#import re
import numpy as np
import pandas as pd
from nltk import word_tokenize
from plistlib import InvalidFileException

from lib.eval import WER
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
        
        #raise NotImplemented("Model evaluation not Implemented!")
        #  - evaluate, print or return the desired outcome
        #

        input = " ".join(test_df["words"])
        output = self.compute(input)

        states_original = " ".join([ state for state in test_df["states"] if state != "" ])
        states_predicted = " ".join([ " ".join([ token["state_token"] for token in line ]) for line in output ])

        wer_score = WER(states_original,states_predicted)

        log.info("WER Score = {} %".format(wer_score))
        log.info("WAcc Score = {} %".format(100-wer_score))

        #
        
    
    #def process_batch(self, input: List[str]):
    #    input = self.viterbi_decoder.__pre_process_batch(input)
    #    (initial_scores, transition_scores, final_scores, emission_scores) = self.viterbi_decoder.compute_scores(input)
    #    y = self.viterbi_decoder.viterbi_decode(input, initial_scores, transition_scores, final_scores, emission_scores)
    #    output = self.__compute_output(y)
    #    return output
    #




    
    # Private methods

    def __pre_process_observation(self, input: str) -> np.ndarray:

        #sequence = [[]] # computed in batches / lines
        
        #raise NotImplemented("You need to preprocess input into suitable format.")
        #
        #rx = re.compile(r'([.()!()?()"()\-()_():(),();()+()*()\[()\]()=()%()€(){()}()«()»()$()`()\\()/()\'])')
        input = input.lower().split("\n")
        _sequence = []
        for line in input:
            #line_sequence = rx.sub(" \\1 ", line)
            #line_sequence = line_sequence.split(" ")
            #line_sequence = [ w for w in line_sequence if w != "" ]
            line_sequence = word_tokenize(line)
            _sequence.append(line_sequence)
        #

        if self.decoder.model_data.with_tokenizer:
            #raise NotImplemented()
            #

            obs_tokens = self.decoder.model_data.tokenizer.obs
            total_words = []
            for line in _sequence:
                total_words.extend(line)
            words_unique = list(set(total_words))

            if not all(w in obs_tokens for w in total_words ):
                words_dict = { words_unique[i] : i for i in range(len(words_unique)) }

            sequence = []
            for line in _sequence:
                if all(w in obs_tokens for w in line):
                    line_sequence = [ obs_tokens[w] for w in line ]
                else:
                    line_sequence = [ obs_tokens[w] if w in obs_tokens else words_dict[w] for w in line ]
                    unknown_tokens = [ w for w in line if not w in obs_tokens ]
                    log.warning("Unkown Tokens '" + str(unknown_tokens) + "' found! Predictions won't be as accurate.")
                    #log.warning("Unkown Tokens found! Predictions won't be as accurate.")
                sequence.append(line_sequence)

            #

        sequence = np.array(sequence, dtype="object")

        return sequence

    @classmethod
    def __compute_output(self, input, decoded_prediction: np.ndarray, tokenizer = False):
        """
        input -> is the input
        decoded_prediction -> is the decoded states to apply into the input

        --
        
        output -> is the output of the state applied into the input
        """
        
        #  - compute and return the predicted states
        #raise NotImplemented("Not Implemented!")

        #if tokenizer != False:
        #    raise NotImplemented("Use tokenizer to resolve tokens from the prediction.")

        #

        state_tokens = dict([(value, key) for key, value in tokenizer.states.items()])
        tokenized_predictions = []
        for line in decoded_prediction:
        #    output_line = " ".join([ state_tokens[prediction] for prediction in line ])
            output_line = [ state_tokens[prediction] for prediction in line ]
            tokenized_predictions.append(output_line)
        #"\n".join(tokenized_predictions)
        
        #rx = re.compile(r'([.()!()?()"()\-()_():(),();()+()*()\[()\]()=()%()€(){()}()«()»()$()`()\\()/()\'])')
        input = input.split("\n")
        _sequence = []
        for line in input:
            #line_sequence = rx.sub(" \\1 ", line)
            #line_sequence = line_sequence.split(" ")
            #line_sequence = [ w for w in line_sequence if w != "" ]
            line_sequence = word_tokenize(line)
            _sequence.append(line_sequence)

        output = []
        for i in range(len(input)):
            _output = [ { 
                    "obs_token": _sequence[i][j],
                    "state_token": tokenized_predictions[i][j]
                }
                for j in range(len(_sequence[i]))
            ]
            output.append(_output)

        #


        return output
