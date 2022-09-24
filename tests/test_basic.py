import pytest
# from analyticsdf.analyticsdataframe import AnalyticsDataframe
import sys
sys.path.insert(0, '/Users/eliwang/DataScience/analytics-dataset/src/analyticsdf')
# sys.path.insert(0, '../src/analyticsdf')
from analyticsdataframe import AnalyticsDataframe
import numpy as np

# initiate an AnalyticsDataframe with default name
def test_init():
    adf = AnalyticsDataframe(100, 3)
    assert adf.predictor_matrix.columns.values.tolist() == ["X1", "X2", "X3"]
    assert np.isnan(adf.predictor_matrix["X1"][0])
    assert np.isnan(adf.response_vector[0])

def test_specify_predictor_name():
    adf = AnalyticsDataframe(5, 3, ["xx1", "xx2", "xx3"], "yy")
    assert adf.predictor_matrix.columns.values.tolist() == ["xx1", "xx2", "xx3"]


# validate statistical properties of generated predictor matrix
def test_update_normal():
    ad = AnalyticsDataframe(10000, 3, ["xx1", "xx2", "xx3"], "yy")
    covariance_matrix = np.array([[1, -0.5, 0.3],
                [-0.5, 1, 0.2],
                [0.3, 0.2, 1]])
    ad.update_predictor_normal(predictor_name_list=["xx1", "xx2", "xx3"],
                                mean=[1, 2, 5],
                                covariance_matrix=covariance_matrix)
    mean_list = list(ad.predictor_matrix.mean())
    corr_list = ad.predictor_matrix.corr()
    assert [round(item, 1) for item in mean_list] == [1.0, 2.0, 5.0]
    assert round(corr_list["xx1"][1], 1) == -0.5
    assert round(corr_list["xx1"][2], 1) == 0.3

## Test handling of variable lists
def test_predictor_normal_handling():
    ad = AnalyticsDataframe(100, 3)
    C = np.array([[1, -0.5, 0.3],
                [-0.5, 1, 0.2],
                [0.3, 0.2, 1]])
    with pytest.raises(Exception):
        ad.update_predictor_normal(predictor_name_list=["xx1", "xx2", "xx3"],
                            mean=[1, 2, 5],
                            covariance_matrix=C)

    # length of mean and length of predictor name list is different
    with pytest.raises(ValueError):
        ad.update_predictor_normal(predictor_name_list=["X1", "X2", "X3"],
                                mean=[1, 2],
                                covariance_matrix=C)

    # shape of covariance matrix is unmatched with length of predictor_name_list
    C = np.array([[1, -0.5],
                [-0.5, 1]])
    with pytest.raises(ValueError):
        ad.update_predictor_normal(predictor_name_list=["X1", "X2", "X3"],
                                    mean=[1, 2, 5],
                                    covariance_matrix=C)

# # Test case: update_predictor_beta
# #
# # ad2 = ad.AnalyticsDataframe(10000, 3, ["xx1", "xx2", "xx3"], "yy")
# # ad2.update_predictor_beta(predictor_name_list = ["xx1", "xx2"],
# #                           a = [1, 2],
# #                           b = [5, 6])

# # Test case - update predictor normal
# #
# ad2 = ad.AnalyticsDataframe(5, 3, ["xx1", "xx2", "xx3"], "yy")
# ad2 = ad.AnalyticsDataframe(10000, 3, ["xx1", "xx2", "xx3"], "yy")
# C = np.array([[1, -0.5, 0.3],
#               [-0.5, 1, 0.2],
#               [0.3, 0.2, 1]])
# ad2.update_predictor_normal(predictor_name_list=["xx1", "xx2", "xx3"],
#                             mean=[1, 2, 5],
#                             covariance_matrix=C)
# print(ad2.response_vector)
# beta = [5, 1, 2, 3]
# eps_var = 1
# pred_list = ["xx1", "xx2", "xx3"]
# ad2.generate_response_vector_linear(beta, eps_var, pred_list)
# print(ad2.response_vector)

# ##
# ## Test case - Update categorical values
# ##
# # ad2 = AnalyticsDataframe(100, 3, ["xx1", "xx2", "xx3"], "yy")
# # print(ad2.predictor_matrix)
# # ad2.update_predictor_categorical("xx2", ["Red", "Green", "Blue"], [0.2, 0.3, 0.5])
# # print(ad2.predictor_matrix)
