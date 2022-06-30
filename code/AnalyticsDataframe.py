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
        populate it with random samples from a uniform distribution over [0, 1).

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
        self.predictor_matrix = DataFrame(np.random.rand(self.n, self.p), columns=self.predictor_names)

        # default response name "Y"
        if response_vector_name is None and self.p:
            response_vector_name = "Y"
        self.response_vector_name = response_vector_name
        self.response_vector = Series(np.random.rand(self.n), name=self.response_vector_name)

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
#           X1        X2        X3        X4        X5
# 0   0.107469  0.935932  0.623685  0.356701  0.068561
# 1   0.256083  0.176947  0.664749  0.999356  0.032216
# 2   0.850852  0.836145  0.177883  0.229526  0.012871
# 3   0.208060  0.004082  0.842855  0.948733  0.629501
# 4   0.277969  0.309295  0.499653  0.951978  0.432877
# ..       ...       ...       ...       ...       ...
# 95  0.379744  0.849693  0.430577  0.061092  0.583688
# 96  0.116081  0.539185  0.592007  0.050244  0.320626
# 97  0.073628  0.928103  0.299621  0.400061  0.065094
# 98  0.222534  0.555966  0.559410  0.044951  0.016164
# 99  0.179221  0.532509  0.948731  0.036875  0.282074
#
# [100 rows x 5 columns]

print(ad1.response_vector)
# 0     0.062972
# 1     0.508997
# 2     0.091036
# 3     0.797928
# 4     0.090736
#         ...
# 95    0.114458
# 96    0.940943
# 97    0.980038
# 98    0.955411
# 99    0.709068
# Name: Y, Length: 100, dtype: float64

ad2 = AnalyticsDataframe(5, 3, ["xx1", "xx2", "xx3"], "yy")
print(ad2.predictor_matrix)
#         xx1       xx2       xx3
# 0  0.282465  0.268106  0.506306
# 1  0.654078  0.706370  0.110114
# 2  0.922589  0.873792  0.071643
# 3  0.135106  0.297225  0.859028
# 4  0.894569  0.270150  0.416135

C = np.array([[1, -0.5, 0.3],
              [-0.5, 1, 0.2],
              [0.3, 0.2, 1]])
ad2.update_predictor_normal(predictor_name_list=["xx1", "xx2", "xx3"],
                            mean_dict={"xx1": 1, "xx2": 2, "xx3": 5},
                            std_dict={"xx1": 0.5, "xx2": 1, "xx3": 1},
                            correlation_matrix=C)
print(ad2.predictor_matrix)
#         xx1       xx2       xx3
# 0  1.725309  0.235418  5.750566
# 1  1.576663 -0.403526  5.401692
# 2  1.230314  0.544714  5.885972
# 3  0.295936  0.712885  4.858719
# 4  1.457847  0.530003  4.571697
