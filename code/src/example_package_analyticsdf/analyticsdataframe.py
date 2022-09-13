import numpy as np
from pandas import Series, DataFrame
from sklearn.utils.extmath import safe_sparse_dot


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
        self.predictor_matrix = DataFrame(np.full([self.n, self.p], np.nan),
                                          columns=self.predictor_names)  # Use numpy.full to set all values to NaN. Can be replaced
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
        :param mean: A numpy array or list, containing mean values
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

        def _is_subset_list(user_input, input_name, actual_list):
            if not set(user_input) <= set(actual_list):
                raise Exception(f'Please select the following: {actual_list} for {input_name}')
            return True
        
        def _is_len_match_list(list1, list1_name, list2, list2_name):
            if not len(list1) == len(list2):
                raise ValueError(f'{list1_name} and {list2_name} must have same length')
            return True

        col_name = self.predictor_matrix.columns.values.tolist()
        if _is_subset_list(predictor_name_list, "predictor_name_list", col_name) and \
            _is_len_match_list(predictor_name_list, "predictor", beta[1:], "beta"):
            eps = epsilon_variance * np.random.randn(self.n)
            beta = np.array(beta)
            if not predictor_name_list:
                predictor_name_list = self.predictor_matrix.columns.values.tolist()
            self.response_vector = safe_sparse_dot(self.predictor_matrix[predictor_name_list],
                                                    beta[1:].T, dense_output=True) + beta[0] + eps
