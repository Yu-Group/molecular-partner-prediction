import numpy as np
import os
from os.path import join as oj
import numpy as np
import mat4py
import pandas as pd
import data
import models
import pickle as pkl
from style import *


def test_pipeline():
    '''test that the data pipeline succesfully completes
    '''
    print('testing pipeline...')
    df = data.get_data()
    assert(df.lifetime.max() < 300)
    
if __name__ == '__main__':
    test_pipeline()
    print('all tests passed!')
    