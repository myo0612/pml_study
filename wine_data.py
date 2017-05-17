import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import plot_decision_regions as pdr
import matplotlib.pyplot as plt

class WineDataSets():

    def __init__(self):
        self.df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header = None)

        self.df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Poanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
        self.X, self.y = self.df_wine.iloc[:, 1:].values, self.df_wine.iloc[:, 0].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.3, random_state = 0)
        mms = MinMaxScaler()
        stdsc = StandardScaler()
        self.X_train_norm = mms.fit_transform(self.X_train)
        self.X_test_norm = mms.transform(self.X_test)
        self.X_train_std = stdsc.fit_transform(self.X_train)
        self.X_test_std = stdsc.transform(self.X_test)

