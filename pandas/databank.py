import pandas as pd

basic_series = pd.Series(['a','b','c','d'])
shape_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=int)
basic_dataframe = pd.DataFrame(
        [[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])

df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
s = df['sepal_width']

modified_index_s = s.copy()
modified_index_s.index = [f'{x}th_index' for x in range(150)]

modified_iris_dataframe = df.copy()
modified_iris_dataframe.iloc[2:7, 0] = None
modified_iris_dataframe.iloc[32:82, 2] = None
modified_iris_dataframe.iloc[140:150, 3] = None

left_half_dataframe = df.iloc[:, 0:2]
right_half_dataframe = df.iloc[:, 2:]
bottom_half_dataframe = df.iloc[75:, :]
top_half_dataframe = df.iloc[:75, :]
