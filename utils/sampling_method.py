# coding:utf-8
import os, sys, time
import pandas as pd

def uniform_subsampling(dataframe, num_point, random_seed=0):
    """
    @dataframe original dataframe that contains (x, y, z, intensity)
    @num_point Number of sampling point
    @return dataframe
    """
    if not isinstance(dataframe, pd.core.frame.DataFrame):
        raise NotImplementedError()

    df = dataframe.sample(n=num_point, random_state=random_seed).reset_index(drop=True)

    return df
