import pytest
from analyticsdf.analyticsdataframe import AnalyticsDataframe

import numpy as np

## Initiate an AnalyticsDataframe with default name
def test_init():
    adf = AnalyticsDataframe(100, 3)
    assert adf.predictor_matrix.columns.values.tolist() == ["X1", "X2", "X3"]
    assert np.isnan(adf.predictor_matrix["X1"][0])
    assert np.isnan(adf.response_vector[0])

def test_specify_predictor_name():
    adf = AnalyticsDataframe(5, 3, ["xx1", "xx2", "xx3"], "yy")
    assert adf.predictor_matrix.columns.values.tolist() == ["xx1", "xx2", "xx3"]


## Validate statistical properties of generated predictor matrix for normal distribution
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

## Test handling of variable lists              ###??? What does it mean by 'variable lists'?
def test_predictor_normal_handling():           ###??? Why don't we put all the update_predictor_normal tests in one function?
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

## Test 'update_predictor_beta'
def test_update_beta():
    ad = AnalyticsDataframe(10000, 3, ["xx1", "xx2", "xx3"], "yy")
    ad.update_predictor_beta(predictor_name_list = ["xx1", "xx2"],
                            a = [1, 2],
                            b = [5, 6])
    pred_matrix = ad.predictor_matrix
    sample_means = pred_matrix.mean().tolist()[:2]
    sample_vars = np.var(pred_matrix).tolist()[:2]
    round_decimals = 2
    xx1_mean = round((1/(1+5)), round_decimals)
    xx2_mean = round((2/(2+6)), round_decimals)
    xx1_var = round((1*5)/(((1+5)**2) * (1+5+1)), round_decimals)
    xx2_var = round((2*6)/(((2+6)**2) * (2+6+1)), round_decimals)
    assert np.isnan(pred_matrix['xx3'][0]) # nan is of type <class 'numpy.float64'>, has to use isnan().
    assert [round(sample_mean, round_decimals) for sample_mean in sample_means] == [xx1_mean, xx2_mean]  # same at 3 decimal points
    assert [round(sample_var, round_decimals) for sample_var in sample_vars] == [xx1_var, xx2_var]



## Test 'update_predictor_categorical'
def test_update_categorical():
    ad = AnalyticsDataframe(1000, 3, ["xx1", "xx2", "xx3"], "yy")
    ad.update_predictor_categorical("xx2", ["Red", "Green", "Blue"], [0.2, 0.3, 0.5])
    pred_matrix = ad.predictor_matrix

    # Test if column 'xx2' has all the assigned values (colors)
    assert sorted(pred_matrix['xx2'].unique()) == sorted(np.array(["Red", "Green", "Blue"]))
    # Test if the probabilities are correct
    prob_red = int(pred_matrix['xx2'].value_counts()['Red']) / 1000      # use int() to convert <numpy.int64>
    prob_green = int(pred_matrix['xx2'].value_counts()['Green']) / 1000
    prob_blue = int(pred_matrix['xx2'].value_counts()['Blue']) / 1000
    probs = [prob_red, prob_green, prob_blue]
    round_decimals = 1
    assert [round(prob, round_decimals) for prob in probs] == [0.2, 0.3, 0.5]

    ## Test its error cases
    # pred_exists error: predictor name doesn't exist
    with pytest.raises(ValueError):
        ad.update_predictor_categorical(predictor_name = "Sunny or Cloudy", 
                                        category_names = ["Red", "Green", "Blue"], 
                                        prob_vector = [0.2, 0.3, 0.5])

    # catg_prob_match error: #category names and #probabiliteis aren't equal
    with pytest.raises(ValueError):
        ad.update_predictor_categorical(predictor_name = "xx1", 
                                        category_names = ["Red", "Green", "Blue"], 
                                        prob_vector = [0.2, 0.3])               
    
    # is_sum_one error: the sum of the probabilities doesn't equal to 1
    with pytest.raises(ValueError):
        ad.update_predictor_categorical(predictor_name = "xx1", 
                                        category_names = ["Red", "Green", "Blue"], 
                                        prob_vector = [0.2, 0.3, 0.9])       


## Test 'generate_response_vector_linear'
def test_response_linear():
    # Initialize the dataframe
    ad = AnalyticsDataframe(1000, 3, ["xx1", "xx2", "xx3"], "yy")
    # Update predictor 'xx1' to normal distribution
    variance = np.array([[0.25]])
    ad.update_predictor_normal(predictor_name_list = ['xx1'], 
                               mean = [2],
                               covariance_matrix = variance)
    # Update predictors 'xx2' and 'xx3' to beta distribution
    ad.update_predictor_beta(predictor_name_list = ["xx2", "xx3"],
                             a = [1, 2],
                             b = [5, 6])
    # Define parameters
    beta = [5, 1, 2, 3]
    eps_var = 1
    pred_list = ["xx1", "xx2", "xx3"]
    ad.generate_response_vector_linear(beta, eps_var, pred_list)

    # Test if first row follows the expected linear regression
    first_row = ad.predictor_matrix.iloc[0, :].tolist() # the first row of 
    np.random.seed(0)                                   # set the seed to make sure the error is the same as in the response generating function
    error = float(eps_var * np.random.randn(1))         
    first_response = beta[0] + beta[1]*first_row[0] + beta[2]*first_row[1] + beta[3]*first_row[2] + error
    assert round(first_response, 2) == round(float(ad.response_vector[0]), 2)