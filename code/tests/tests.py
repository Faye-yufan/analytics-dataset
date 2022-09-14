import sys
sys.path.insert(1, '../src')
import example_package_analyticsdf as ad
import numpy as np

# initiate an AnalyticsDataframe with default name
ad1 = ad.AnalyticsDataframe(100, 5)
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

ad2 = ad.AnalyticsDataframe(5, 3, ["xx1", "xx2", "xx3"], "yy")
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
ad2 = ad.AnalyticsDataframe(10000, 3, ["xx1", "xx2", "xx3"], "yy")
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

ad3 = ad.AnalyticsDataframe(100, 3)
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


# Test case: update_predictor_beta
#
# ad2 = ad.AnalyticsDataframe(10000, 3, ["xx1", "xx2", "xx3"], "yy")
# ad2.update_predictor_beta(predictor_name_list = ["xx1", "xx2"],
#                           a = [1, 2],
#                           b = [5, 6])

# Test case - update predictor normal
#
ad2 = ad.AnalyticsDataframe(5, 3, ["xx1", "xx2", "xx3"], "yy")
ad2 = ad.AnalyticsDataframe(10000, 3, ["xx1", "xx2", "xx3"], "yy")
C = np.array([[1, -0.5, 0.3],
              [-0.5, 1, 0.2],
              [0.3, 0.2, 1]])
ad2.update_predictor_normal(predictor_name_list=["xx1", "xx2", "xx3"],
                            mean=[1, 2, 5],
                            covariance_matrix=C)
print(ad2.response_vector)
beta = [5, 1, 2, 3]
eps_var = 1
pred_list = ["xx1", "xx2", "xx3"]
ad2.generate_response_vector_linear(beta, eps_var, pred_list)
print(ad2.response_vector)

##
## Test case - Update categorical values
##
# ad2 = AnalyticsDataframe(100, 3, ["xx1", "xx2", "xx3"], "yy")
# print(ad2.predictor_matrix)
# ad2.update_predictor_categorical("xx2", ["Red", "Green", "Blue"], [0.2, 0.3, 0.5])
# print(ad2.predictor_matrix)
