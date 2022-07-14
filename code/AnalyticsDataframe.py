# Author: Fei
import numpy as np
from pandas import Series, DataFrame


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

    def update_predictor_normal(self, predictor_name_list: list = None,
                                mean: np.ndarray = None,
                                covariance_matrix: np.ndarray = None):
        """
        update_predictor_normal(self, predictor_name_list=None, mean=None,
                                covariance_matrix=None)
        Update the predictors of the instance to normally distributed.
        ------
        :param predictor_name_list: A list of predictor names in the initial AnalyticsDataframe
        :param mean: A numpy array or list,
                        value: mean
        :param covariance_matrix: A symmetric and positive semi-definite N * N matrix,
                                    defines correlation among N variables.
        :return:
        """
        col_nm = self.predictor_matrix.columns.values.tolist()
        if not set(predictor_name_list) <= set(col_nm):
            raise Exception(f'Please select the following predictors: {col_nm}')

        # Check if parameters have the same size
        # for covariance matrix, it shall be symmetric and positive semi-definite
        def _is_len_matched(_name: list, _mean: np.ndarray):
            return len(_name) == len(_mean)

        # Update predictor data by mean and variance
        if _is_len_matched(predictor_name_list, mean) \
                and len(predictor_name_list) > 0:
            num_row = len(self.predictor_matrix)
            self.predictor_matrix[predictor_name_list] = np.random.multivariate_normal(mean,
                                                                                       covariance_matrix,
                                                                                       size=num_row,
                                                                                       check_valid='warn')

        elif not _is_len_matched(predictor_name_list, mean):
            raise ValueError('predictor and mean must have same length')


# initiate an AnalyticsDataframe with default name
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

##
## Test case 1 - validate statistical properties of generated predictor matrix
##
ad2 = AnalyticsDataframe(10000, 3, ["xx1", "xx2", "xx3"], "yy")
C = np.array([[1, -0.5, 0.3],
              [-0.5, 1, 0.2],
              [0.3, 0.2, 1]])
ad2.update_predictor_normal(predictor_name_list=["xx1", "xx2", "xx3"],
                            mean=[1, 2, 5],
                            covariance_matrix=C)
test_matrix_1 = ad2.predictor_matrix
print(test_matrix_1.mean())
# xx1    1.003769
# xx2    2.007161
# xx3    5.008807

print(test_matrix_1.corr())
#           xx1       xx2       xx3
# xx1  1.000000 -0.501385  0.300073
# xx2 -0.501385  1.000000  0.203265
# xx3  0.300073  0.203265  1.000000

##
## Test case 2 - Test handling of variable lists
##

ad3 = AnalyticsDataframe(100, 3)
C = np.array([[1, -0.5, 0.3],
              [-0.5, 1, 0.2],
              [0.3, 0.2, 1]])


# Should generate error due to mismatch of variable names
#
# ad3.update_predictor_normal(predictor_name_list=["xx1", "xx2", "xx3"],
#                             mean=[1, 2, 5],
#                             covariance_matrix=C)
# ...
# Exception: Please select the following predictors: ['X1', 'X2', 'X3']


# Should generate error due to mismatch of shape between covariance and mean
#
# C2 = np.array([[1, -0.5],
#               [-0.5, 1]])
# ad3.update_predictor_normal(predictor_name_list=["X1", "X2", "X3"],
#                             mean=[1, 2, 5],
#                             covariance_matrix=C2)
# ...
# ValueError: mean and cov must have same length


# Should generate error due to mismatch of shape between predictor and mean
#
# ad3.update_predictor_normal(predictor_name_list=["X1", "X2", "X3"],
#                             mean=[1, 2],
#                             covariance_matrix=C)
# ...
# ValueError: predictor and mean must have same length