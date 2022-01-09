import torch
import os
import logging
import traceback
def load_model(path,Model_Structure,model_init_params):
    model=None
    try:
        checkpoint=torch.load(path)
        model=Model_Structure(**model_init_params)
        model.load_state_dict(checkpoint["model_state_dict"])
        logging.info("Load model from "+path)
    except:
        traceback.print_exc()
    return model 