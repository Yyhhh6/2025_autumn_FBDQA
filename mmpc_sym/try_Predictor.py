import pandas as pd
from .Predictor import Predictor

path = "data/data_try_Predictor/snapshot_sym2_date0_am.csv"

# 读取为 DataFrame
x = pd.read_csv(path)

# print(type(x))
# print(x.head())

pred = Predictor()
pred.predict([x])