import pandas as pd
from joblib import dump, load
import numpy as np


class Predictor:
    def __init__(self, path):
        print("reading dataset")
        self.dataset = pd.read_csv(path, sep=',', header=0)
        self.target = "% Silica Concentrate"
        self.mean = {}
        self.models = {}
        self.model_names = ['xgbr', 'rf', 'cb']

    def preprocess(self):
        print("preprocessing")
        to_drop = ['Ore Pulp pH', 'Flotation Column 01 Air Flow', 'Flotation Column 02 Air Flow',
                   'Flotation Column 03 Air Flow', 'date']
        self.dataset = self.dataset.drop(to_drop, axis=1)
        self.dataset = self.dataset.drop(self.target, axis=1)

        for feat in self.dataset.columns:
            if type(self.dataset[feat][0]) == str:
                self.dataset[feat] = self.dataset[feat].apply(self.convert)
            else:
                self.dataset[feat] = self.dataset[feat].astype(float)

        for col in self.dataset.columns:
            self.mean[col] = self.dataset[col].mean()

    def load_models(self):
        print("loading models")
        for name in self.model_names:
            print("#", end='')
            self.models[name] = load(f"models/{name}.joblib")
        print("\ndone")

    def convert(self, x):
        x = x.replace(',', '.')
        return float(x)

    def get_default(self, feat):
        return self.mean[feat]

    def get_cols(self):
        return self.dataset.columns

    def predict(self, params):
        print(f"params {params}")
        for col in self.dataset.columns:
            if col not in params or params[col] == '':
                params[col] = self.get_default(col)

        series = pd.Series(params)
        ans = {}
        print(f"got: {series}")
        for name, model in self.models.items():
            if name == "xgbr":
                series = series.to_numpy()
            ans[name] = model.predict(np.array(series).reshape(-1, 1).T)
        return ans
