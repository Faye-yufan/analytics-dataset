import numpy as np
from pandas import Series, DataFrame
from sklearn.utils.extmath import safe_sparse_dot

from example_package_analyticsdf.utils import check_update_normal, check_update_beta, check_response_linear

class AnalyticsDataframe:
    """Create a AnalyticsDataframe class.

    Creates a dataframe class which uses the ``n``, ``p``, ``predictor_names`` 
    and ``response_vector_name`` arguments to initialize a dataframe.

    Args:
        n:
            Number of observations.
        p:
            Number of predictors.
        predictor_names:
            List of strings (default = [`X1`, `X2`, … `Xp`]).
        response_vector_name:
            String (default = `Y`).

    Returns:
        AnalyticsDataframe class:
            predictor_matrix: a Pandas Dataframe with Nan.
            response_vector:  a Pandas Series with Nan.
    """

    def __init__(self, n, p,
                 predictor_names=None,
                 response_vector_name=None):
        self.n = n
        self.p = p

        if predictor_names is None and self.p:
            predictor_names = ["X{}".format(x) for x in list(range(1, self.p + 1))]
        self.predictor_names = predictor_names
        self.predictor_matrix = DataFrame(np.full([self.n, self.p], np.nan),
                                          columns=self.predictor_names)  

        if response_vector_name is None and self.p:
            response_vector_name = "Y"
        self.response_vector_name = response_vector_name
        self.response_vector = Series(np.full([self.n], np.nan), name=self.response_vector_name)

    @check_update_normal
    def update_predictor_normal(self, predictor_name_list: list = None,
                                mean: np.ndarray = None,
                                covariance_matrix: np.ndarray = None):
        """Update the predictors of the instance to normally distributed.

        Args:
            predictor_name_list:
                A list of predictor names in the initial AnalyticsDataframe.
            mean:
                A numpy array or list, containing mean values.
            covariance_matrix:
                A symmetric and positive semi-definite N * N matrix, defines correlation among N variables.

        """
        num_row = len(self.predictor_matrix)
        self.predictor_matrix[predictor_name_list] = np.random.multivariate_normal(mean,
                                                                                    covariance_matrix,
                                                                                    size=num_row,
                                                                                    check_valid='warn')

    @check_update_beta
    def update_predictor_beta(self, a, b, predictor_name_list: list = None):
        """
        update_predictor_beta(self, predictor_name_list: list = None, a, b)
        Update predictor as beta distributed
        :param predictor_name_list: A list of predictor names in the initial AnalyticsDataframe
        :param a: float or array_like of floats. Alpha, positive (>0).
        :param b: float or array_like of floats. Beta, positive (>0).
        :return:
        """
        num_row = len(self.predictor_matrix)
        pred_nparr = np.random.beta(a, b, (1, num_row, len(predictor_name_list)))
        pred_pds = pred_nparr.reshape(num_row, len(predictor_name_list))
        self.predictor_matrix[predictor_name_list] = pred_pds


    def update_predictor_categorical(self, predictor_name = None,
                                     category_names: list = None,
                                     prob_vector: np.array = None):
        """
        update_predictor_categorical(self, predictor_name=None, category_names=None, prob_vector=None)
        Update a predictor with categorical values.
        -----
        :param predictor_name: A string that contains the name of the variable to be set
        :param category_names: A vector of strings that contains names of the different category values
        :param prob_vector: A vector of numerics of the same length as category_names that specifies the 
                            probability (frequency) of each category value.
        :return:
        """
        ## Create a function to check if the target predictor exists
        def pred_exists(name):
            """
            Check if the target predictor exists.
            """
            return name in self.predictor_names

        ## Create a function to check if the number of category names and probabilities equal
        def catg_prob_match(names, vector):
            """
            Check if the number of category names and the length of probability vector equal. 
            Aiming to ensure each category has its own and only probability.
            """
            return len(names) == len(vector)

        ## Create a function to check if the sum of the probabilities is 1
        def is_sum_one(vector):
            """
            Check if the sum of the probabilities is 1.
            """
            return sum(vector) == 1

        if pred_exists(predictor_name) and catg_prob_match(category_names, 
           prob_vector) and is_sum_one(prob_vector):
            catg_dict = {} # key is 0, 1, 2,...; value is the corresponding category name
            num = len(category_names)
            for i in range(num): # i is 0, 1, 2,...
                catg_dict[i] = category_names[i]
            self.predictor_matrix[predictor_name] = np.random.choice(
                                                    a = list(catg_dict.keys()),
                                                    size = len(self.predictor_matrix[predictor_name]),
                                                    p = prob_vector)
            # Convert keys (0, 1, 2,...) to actual categories
            df = self.predictor_matrix
            nrow = len(df[predictor_name])                                      
            for j in range(nrow):
                # value = self.predictor_matrix[predictor_name][j]
                # self.predictor_matrix[predictor_name][j] = catg_dict[value]  # Avoid chained indexing
                value = df.loc[df.index[j], predictor_name]
                df.loc[df.index[j], predictor_name] = catg_dict[value]

        elif not pred_exists(predictor_name):
            raise ValueError('Please choose one of the existing predictors!')

        elif not catg_prob_match(category_names, prob_vector):
            raise ValueError("Probabilities should have the same amount as categories!")
        
        elif not is_sum_one(prob_vector):
            raise ValueError("The sum of probabilities should equal to 1!")

    @check_response_linear
    def generate_response_vector_linear(self, beta: list = None,
                                        epsilon_variance: float = None,
                                        predictor_name_list: list = None):
        """
        generate_response_vector_linear(self, beta: list = None,
                                        epsilon_variance: float = None,
                                        predictor_name_list: list = None):
        Generates a response vector based on a linear regression generative model
        ------
        :param beta: A list, coefficients of the linear model – first coefficient is the intercept
        :param epsilon_variance: a scalar variance specification
        :param predictor_name_list: A list of predictor names to be applied with this method
        :return:
        """

        eps = epsilon_variance * np.random.randn(self.n)
        beta = np.array(beta)
        if not predictor_name_list:
            predictor_name_list = self.predictor_matrix.columns.values.tolist()
        self.response_vector = safe_sparse_dot(self.predictor_matrix[predictor_name_list],
                                                beta[1:].T, dense_output=True) + beta[0] + eps
