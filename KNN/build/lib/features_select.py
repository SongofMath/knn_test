import knnManager as knn
import pandas as pd
import graphManager as gm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold


def var_columns(data):
    selector = VarianceThreshold()
    selector.fit_transform(data)
    return selector.variances_


