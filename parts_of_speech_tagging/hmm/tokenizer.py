
class Tokenizer:

    obs: dict()
    states: dict()

    def __init__(self,x_dict,y_dict):
        self.obs = x_dict
        self.states = y_dict
    
