import random
import pytest
from analyticsdf.analyticsdataframe import AnalyticsDataframe
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

    tolerance = 0.1                         # because the beta distribution generated has randomness, we allow some deviation from the theoretical value
    assert np.isnan(pred_matrix['xx3'][0])  # nan is of type <class 'numpy.float64'>, has to use isnan().
    assert [round(sample_mean, round_decimals) for sample_mean in sample_means] >= [(1 - tolerance) * xx1_mean, (1 - tolerance) * xx2_mean]
    assert [round(sample_var, round_decimals) for sample_var in sample_vars] >= [(1 - tolerance) * xx1_var, (1 - tolerance) * xx2_var]
    assert [round(sample_mean, round_decimals) for sample_mean in sample_means] <= [(1 + tolerance) * xx1_mean, (1 + tolerance) * xx2_mean]
    assert [round(sample_var, round_decimals) for sample_var in sample_vars] <= [(1 + tolerance) * xx1_var, (1 + tolerance) * xx2_var]


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
    ad.generate_response_vector_linear(beta=beta, epsilon_variance=eps_var, predictor_name_list=pred_list)

    # Test if first row follows the expected linear regression
    first_row = ad.predictor_matrix.iloc[0, :].tolist() # the first row of the predicted values
    error = float(eps_var * np.random.randn(1))         
    first_response = beta[0] + beta[1]*first_row[0] + beta[2]*first_row[1] + beta[3]*first_row[2] + error

    # Need a better solution than setting high tolerance!
    tolerance = 0.5
    assert first_response >= float((1 - tolerance) * ad.response_vector[0])
    assert first_response <= float((1 + tolerance) * ad.response_vector[0])

    ## Test its error cases
    # _is_subset_list error: at least one input predictor is not in the data columns
    with pytest.raises(Exception):
        ad.generate_response_vector_linear(beta=beta, epsilon_variance=eps_var, predictor_name_list=['xx1', 'You got this mate'])
    # _is_len_match_list error: #input predictors + 1 must equal to #betas
    with pytest.raises(ValueError):
        ad.generate_response_vector_linear(beta=[5, 1, 2], epsilon_variance=eps_var, predictor_name_list=pred_list)

# Test 'generate_response_vector_polynomial'
def test_response_polynomial():
    ad = AnalyticsDataframe(100, 6)
    C = np.array([[1, -0.5, 0.3],
                [-0.5, 1, 0.2],
                [0.3, 0.2, 1]])
    ad.update_predictor_normal(predictor_name_list=["X1", "X2", "X4"],
                                mean=[1, 2, 5],
                                covariance_matrix=C)
    
    predictor_name_list = ["X1", "X2", "X4"]
    polynomial_order = [1, 2, 3]
    beta = [1.5, 3, -2, 0.5, 2, 0, 0.1]
    int_matrix = np.array([
        [0,0,0,0,0,0], 
        [-0.5,0,0,0,0,0], 
        [0,0,0,0,0,0], 
        [0,0,0,0,0,0], 
        [0,0,0,0,0,0], 
        [0,-3,0,0,0,0]])
    eps_var = 1

    ad.generate_response_vector_polynomial(
                predictor_name_list = predictor_name_list, 
                polynomial_order = polynomial_order, 
                beta = beta,
                interaction_term_betas = int_matrix, 
                epsilon_variance = eps_var)
    
    # Test if a random row follows the expected linear regression
    i = random.randint(0, 99)
    row = ad.predictor_matrix.iloc[i, :].tolist() # the first row of the predicted values
    print(row)
    error = float(eps_var * np.random.randn(1))         
    response = beta[0] + beta[1] * row[0] ** 1 + \
            beta[2] * row[1] ** 1 + beta[3] * row[1] ** 2 + \
            beta[4] * row[3] ** 1 + beta[5] * row[3] ** 2 + beta[6] * row[3] ** 3 + \
            int_matrix[1][0] * row[0] * row[1] + int_matrix[5][1] * row[1] * row[3] ** 3 +error
    print(int_matrix[1][0], int_matrix[5][1])

    # Need a better solution than setting high tolerance!
    tolerance = 0.5
    assert (ad.response_vector[i] - error) <= response <= (ad.response_vector[i] + error)
    
    with pytest.raises(KeyError):
        ad.generate_response_vector_polynomial(
                predictor_name_list = ['xx1', 'You got this mate'], 
                polynomial_order = polynomial_order, 
                beta = beta,
                interaction_term_betas = int_matrix, 
                epsilon_variance = 1)