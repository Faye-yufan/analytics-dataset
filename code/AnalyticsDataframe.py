# Author: Fei
import numpy as np
from pandas import Series, DataFrame
from scipy.linalg import cholesky


class AnalyticsDataframe:
    """
    Instantiation parameters:
        n:  number of observations
        p:  number of predictors
        predictor_names:  list of strings (default = [‘X1’, ‘X2’, … ‘Xp’])
        response_vector_name:  string (default = ‘Y’)

    Data:
        response_vector:  Pandas Series
        predictor_matrix:  Pandas Dataframe
        ------
        Use numpy.random.rand to Create an array of the given shape and
        set the default values to NaN.

    """

    def __init__(self, n, p,
                 predictor_names=None,
                 response_vector_name=None):
        self.n = n
        self.p = p

        # If did not define predictor_names, default names [‘X1’, ‘X2’, … ‘Xp’]
        if predictor_names is None and self.p:
            predictor_names = ["X{}".format(x) for x in list(range(1, self.p + 1))]
        self.predictor_names = predictor_names
        self.predictor_matrix = DataFrame(np.full([self.n, self.p], np.nan), columns=self.predictor_names)  # Use numpy.full to set all values to NaN. Can be replaced
                                                                                                            # by any other value.

        # default response name "Y"
        if response_vector_name is None and self.p:
            response_vector_name = "Y"
        self.response_vector_name = response_vector_name
        self.response_vector = Series(np.full([self.n], np.nan), name=self.response_vector_name)

    def update_predictor_normal(self, predictor_name_list: list = None, mean_dict: dict = None,
                                std_dict: dict = None, correlation_matrix: np.ndarray = None):
        """
        update_predictor_normal(self, predictor_name_list=None, mean_dict=None,
                                std_dict=None, covariance_matrix=None)

        Update the predictors of the instance to normally distributed.
        ------
        :param predictor_name_list: A list of predictor names in the initial AnalyticsDataframe
        :param mean_dict: A dictionary,
                        key: predictor name,
                        value: mean
        :param std_dict: A dictionary,
                        key: predictor name,
                        value: standard deviation
        :param correlation_matrix: A symmetric N * N matrix, defines correlation among N variables.
        :return:
        """
        col_nm = self.predictor_matrix.columns.values.tolist()
        if not set(predictor_name_list) <= set(col_nm):
            raise Exception(f'Please select the following predictors: {col_nm}')

        # Check if update parameters have the same predictor name
        def is_name_matched(name: list, mean: dict, var: dict):
            return mean.keys() | set() == set(name) & mean.keys() == var.keys()

        # Update predictor data by mean and variance
        if is_name_matched(predictor_name_list, mean_dict, std_dict) \
                and len(predictor_name_list) > 0:
            for col in predictor_name_list:
                self.predictor_matrix[col] = np.random.normal(mean_dict[col], std_dict[col], self.n)
        elif not is_name_matched(predictor_name_list, mean_dict, std_dict):
            raise Exception(f'Unmatched inputs {predictor_name_list}, {mean_dict.keys()}, {std_dict.keys()} ')

        # Use correlated matrix to update predictors
        if correlation_matrix.any():
            # Cholesky decomposition
            c = correlation_matrix
            u = cholesky(c)
            # DataFrame of correlated random sequences
            p_arr = self.predictor_matrix[predictor_name_list].to_numpy()
            pc = p_arr @ u
            self.predictor_matrix[predictor_name_list] = pc


ad1 = AnalyticsDataframe(100, 5)
print(ad1.predictor_matrix)
#     X1  X2  X3  X4  X5
# 0  NaN NaN NaN NaN NaN
# 1  NaN NaN NaN NaN NaN
# 2  NaN NaN NaN NaN NaN
# 3  NaN NaN NaN NaN NaN
# 4  NaN NaN NaN NaN NaN
# ..  ..  ..  ..  ..  ..
# 95 NaN NaN NaN NaN NaN
# 96 NaN NaN NaN NaN NaN
# 97 NaN NaN NaN NaN NaN
# 98 NaN NaN NaN NaN NaN
# 99 NaN NaN NaN NaN NaN
# 
# [100 rows x 5 columns]

print(ad1.response_vector)
# 0    NaN
# 1    NaN
# 2    NaN
# 3    NaN
# 4    NaN
#       ..
# 95   NaN
# 96   NaN
# 97   NaN
# 98   NaN
# 99   NaN
# Name: Y, Length: 100, dtype: float64

ad2 = AnalyticsDataframe(5, 3, ["xx1", "xx2", "xx3"], "yy")
print(ad2.predictor_matrix)
#    xx1  xx2  xx3
# 0  NaN  NaN  NaN
# 1  NaN  NaN  NaN
# 2  NaN  NaN  NaN
# 3  NaN  NaN  NaN
# 4  NaN  NaN  NaN

C = np.array([[1, -0.5, 0.3],
              [-0.5, 1, 0.2],
              [0.3, 0.2, 1]])
ad2.update_predictor_normal(predictor_name_list=["xx1", "xx2", "xx3"],
                            mean_dict={"xx1": 1, "xx2": 2, "xx3": 5},
                            std_dict={"xx1": 0.5, "xx2": 1, "xx3": 1},
                            correlation_matrix=C)
print(ad2.predictor_matrix)
#         xx1       xx2       xx3
# 0  1.865383  2.218858  6.750145
# 1  0.562747  1.535777  6.324234
# 2  1.381225  1.521372  7.069359
# 3  0.937004  1.359714  5.233999
# 4  0.853955  1.408445  5.944211
