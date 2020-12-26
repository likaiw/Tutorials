import pandas as pd
import operator
import numpy as np
import databank as db

df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
s = df['sepal_length']


def is_equal_frame(ans, res):
    return type(ans) is pd.DataFrame and pd.testing.assert_frame_equal(ans, res) is None


def is_equal_series(ans, res):
    return type(ans) is pd.Series and pd.testing.assert_series_equal(ans, res) is None


def is_equal(ans, res):
    return type(ans) is not pd.Series and ans is not pd.DataFrame and operator.eq(ans, res)


def assert_ans_equal_res_with_comment(ans, res):
    if is_equal_frame(ans, res) or is_equal_series(ans, res) or is_equal(ans, res):
        print("""
******************************
***   Correct, good job!   ***
******************************
                """)
    else:
        print("""
*********************************
***   Incorrect, try again!   ***
*********************************
            """)

    print("Your response:")
    print(res)
    print()
    print("Correct answer:")
    print(ans)
    print()


# CONSTRUCTOR

def assert_series_constructor_data_value_1(res):
    ans = pd.Series(['a', 'b', 'c', 'd'], index=[1, 2, 3, 4])
    assert_ans_equal_res_with_comment(ans, res)


def assert_series_constructor_data_value_2(res):
    ans = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
    assert_ans_equal_res_with_comment(ans, res)


def assert_series_constructor_dtype_1(res):
    ans = pd.Series([1, 2, 3, 4], dtype=float)
    assert_ans_equal_res_with_comment(ans, res)


def assert_series_constructor_name_1(res):
    ans = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'], dtype=int, name='MySeries')
    assert_ans_equal_res_with_comment(ans, res)


def assert_series_attribute_shape_1(res):
    ans = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=int).shape
    assert_ans_equal_res_with_comment(ans, res)


def assert_dataframe_constructor_1(res):
    ans = pd.DataFrame(
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]])
    assert_ans_equal_res_with_comment(ans, res)


def assert_dataframe_constructor_2(res):
    ans = pd.DataFrame(
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]],
        columns=['x', 'y', 'z'])
    assert_ans_equal_res_with_comment(ans, res)


# VIEWING

def assert_viewing_1_1(res):
    ans = df.head(10)
    assert_ans_equal_res_with_comment(ans, res)


def assert_viewing_2_1(res):
    ans = df.tail(10)
    assert_ans_equal_res_with_comment(ans, res)


def assert_viewing_3_1(res):
    ans = df.shape
    assert_ans_equal_res_with_comment(ans, res)


def assert_viewing_4_1(res):
    ans = 'object'
    assert_ans_equal_res_with_comment(ans, res)


def assert_viewing_5_1(res):
    ans = 5.1
    assert_ans_equal_res_with_comment(ans, res)


def assert_viewing_6_1(res):
    ans = 4
    assert_ans_equal_res_with_comment(ans, res)


def assert_viewing_7_1(res):
    ans = df.copy()['sepal_length'].apply(np.square)
    assert_ans_equal_res_with_comment(ans, res)


# SELECTING

def assert_selecting_1_1(res):
    ans = df['petal_width']
    assert_ans_equal_res_with_comment(ans, res)


def assert_selecting_2_1(res):
    ans = df[['petal_width', 'petal_length']]
    assert_ans_equal_res_with_comment(ans, res)


def assert_selecting_3_1(res):
    ans = s.iloc[100]
    assert_ans_equal_res_with_comment(ans, res)


def assert_selecting_4_1(res):
    ans = s.copy()
    ans.index = [f'{x}th_index' for x in range(ans.size)]
    ans = ans.loc['100th_index']
    assert_ans_equal_res_with_comment(ans, res)


def assert_selecting_5_1(res):
    ans = df.iloc[15, 3]
    assert_ans_equal_res_with_comment(ans, res)


def assert_selecting_5_2(res):
    ans = df.iloc[3, :]
    assert_ans_equal_res_with_comment(ans, res)


def assert_selecting_5_3(res):
    ans = df.iloc[18:20, 1:3]
    assert_ans_equal_res_with_comment(ans, res)


# CLEANING

na_df = db.modified_iris_dataframe
na_s = na_df['sepal_width']


def assert_cleaning_1_1(res):
    ans = na_df.copy().dropna()
    assert_ans_equal_res_with_comment(ans, res)


def assert_cleaning_1_2(res):
    ans = na_df.copy().dropna(axis=1)
    assert_ans_equal_res_with_comment(ans, res)


def assert_cleaning_1_3(res):
    ans = na_df.copy().dropna(axis=1, thresh=120)
    assert_ans_equal_res_with_comment(ans, res)


def assert_cleaning_2_1(res):
    ans = na_df.copy()['sepal_length'].fillna(s.median())
    assert_ans_equal_res_with_comment(ans, res)


def assert_cleaning_2_2(res):
    ans = na_df.copy().fillna(s.mean())
    assert_ans_equal_res_with_comment(ans, res)


def assert_cleaning_3_1(res):
    ans = na_s.copy().astype(int)
    assert_ans_equal_res_with_comment(ans, res)


def assert_cleaning_4_1(res):
    ans = na_s.copy().replace([3.0, 2.0], ['Three', 'Two'])
    assert_ans_equal_res_with_comment(ans, res)


# FILTER

def assert_filter_1_1(res):
    ans = df[(df['petal_width'] < 1)]
    assert_ans_equal_res_with_comment(ans, res)


def assert_filter_1_2(res):
    ans = df[(df['petal_width'] > 0.5) & (df['petal_width'] < 1)]
    assert_ans_equal_res_with_comment(ans, res)


# SORT

def assert_sort_1_1(res):
    ans = df.copy().sort_values('petal_length')
    assert_ans_equal_res_with_comment(ans, res)


def assert_sort_1_2(res):
    ans = df.copy().sort_values('petal_length', ascending=False)
    assert_ans_equal_res_with_comment(ans, res)


def assert_sort_1_3(res):
    ans = df.copy().sort_values(['petal_length', 'petal_width'], ascending=[True, False])
    assert_ans_equal_res_with_comment(ans, res)


# GROUPBY

def assert_groupby_1_1(res):
    ans = df.copy().groupby('petal_width').mean()
    assert_ans_equal_res_with_comment(ans, res)


def assert_groupby_1_2(res):
    ans = df.copy().groupby(['petal_width', 'petal_length']).max()
    assert_ans_equal_res_with_comment(ans, res)


# APPLY

def assert_apply_1_1(res):
    ans = df.copy().iloc[:, :-1].apply(np.mean)
    assert_ans_equal_res_with_comment(ans, res)


# JOINING

def assert_joining_1_1(res):
    ans = df
    assert_ans_equal_res_with_comment(ans, res)


def assert_joining_1_2(res):
    ans = db.bottom_half_dataframe.append(db.top_half_dataframe)
    assert_ans_equal_res_with_comment(ans, res)


def assert_joining_2_1(res):
    ans = df
    assert_ans_equal_res_with_comment(ans, res)


def assert_joining_2_2(res):
    ans = pd.concat([db.right_half_dataframe, db.left_half_dataframe], axis = 1)
    assert_ans_equal_res_with_comment(ans, res)


# STATISTICS

def assert_statistics_1_1(res):
    ans = df.describe()
    assert_ans_equal_res_with_comment(ans, res)


def assert_statistics_2_1(res):
    ans = df.corr()
    assert_ans_equal_res_with_comment(ans, res)
