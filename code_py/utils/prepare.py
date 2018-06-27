import numpy as np
import sys,os
def prepare_model(model_fold):
    model = dict()
    model['mean_file'] = os.path.join(model_fold,'mean_image.mat')
    model['model_file'] = os.path.join(model_fold,'test.prototxt')
    return model
