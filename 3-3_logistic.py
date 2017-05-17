from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import plot_decision_regions as pdr
import matplotlib.pyplot as plt
import iris_data as ir
import useSkLearn as sk

sk.RunSkMethod("lr")
"""
lr = LogisticRegression(C = 100.0, random_state = 0)
dd = ir.IrisDataSets()
dd.useFit(lr)
pdr.plot_decision_regions(X = dd.X_combined_std, y = dd.y_combined, classifier = lr, test_idx = range(105, 150))
dd.drawGraph()
"""
