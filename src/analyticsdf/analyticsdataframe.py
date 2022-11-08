import numpy as np
from pandas import Series, DataFrame
from sklearn.utils.extmath import safe_sparse_dot
from itertools import combinations

# from analyticsdf.utils import check_update_normal, check_update_beta, check_response_linear
from analyticsdf import check_columns_exist

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

    @check_columns_exist
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
        
        Raises:
            KeyError: If the column does not exists.
            ValueError: If mean and cov does not have the same size.

        """
        num_row = len(self.predictor_matrix)
        self.predictor_matrix[predictor_name_list] = np.random.multivariate_normal(mean,
                                                                                    covariance_matrix,
                                                                                    size=num_row,
                                                                                    check_valid='warn')

    @check_columns_exist
    def update_predictor_beta(self, predictor_name_list, a, b):
        """Update the predictors of the instance as beta distributed.

        Args:
            predictor_name_list:
                A list of predictor names in the initial AnalyticsDataframe.
            a: 
                float or array_like of floats. Alpha, positive (>0).
            b: 
                float or array_like of floats. Beta, positive (>0).
        
        Raises:
            KeyError: If the column does not exists.

        """
        num_row = len(self.predictor_matrix)
        pred_nparr = np.random.beta(a, b, (1, num_row, len(predictor_name_list)))
        pred_pds = pred_nparr.reshape(num_row, len(predictor_name_list))
        self.predictor_matrix[predictor_name_list] = pred_pds


    def update_predictor_categorical(self, predictor_name = None,
                                     category_names: list = None,
                                     prob_vector: np.array = None):
        """Update a predictor with categorical values.

        Args:
            predictor_name:
                A predictor name in the initial AnalyticsDataframe.
            category_names: 
                A vector of strings that contains names of the different category values
            prob_vector: 
                A vector of numerics of the same length as category_names that specifies the probability (frequency) of each category value.
        
        Raises:
            KeyError: If the column does not exists.
            ValueError: If sum of ``prob_vector`` not equal to 1.
            ValueError: If length of ``prob_vector`` not equal to ``category_names``.

        """
        if predictor_name not in self.predictor_names:
            raise ValueError('Please choose one of the existing predictors!')

        if sum(prob_vector) != 1:
            raise ValueError("The sum of probabilities should equal to 1!")

        if len(category_names) != len(prob_vector):
            raise ValueError("Probabilities should have the same amount as categories!")

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


    @check_columns_exist
    def generate_response_vector_linear(self, predictor_name_list: list = None, 
                                        beta: list = None,
                                        epsilon_variance: float = None):
        """Generates a response vector based on a linear regression generative model.

        Args:
            predictor_name_list:
                A list of predictor names in the initial AnalyticsDataframe.
            beta: 
                A list, coefficients of the linear model – first coefficient is the intercept
            epsilon_variance: 
                A scalar variance specification.
        
        Raises:
            KeyError: If the column does not exists.

        """
        eps = epsilon_variance * np.random.randn(self.n)
        beta = np.array(beta)
        if not predictor_name_list:
            predictor_name_list = self.predictor_matrix.columns.values.tolist()
        self.response_vector = safe_sparse_dot(self.predictor_matrix[predictor_name_list],
                                                beta[1:].T, dense_output=True) + beta[0] + eps


    @check_columns_exist
    def generate_response_vector_polynomial(self, predictor_name_list: list,
                                            polynomial_order: list,
                                            beta: list,
                                            interaction_term_betas: np.array,
                                            epsilon_variance: float):

        """Generates a response vector based on a linear regression generative model that contains 
        polynomial terms for one or more of the predictors and interaction terms.

        Args:
            predictor_name_list:
                A list of predictor names in the initial AnalyticsDataframe.
            polynomial_order: 
                A list of integers that specify the order of the polynomial for each predictor with legal values of 1 to 4.
            beta_vector: 
                A list of the betas (coefficients of the linear model) 
                    – First coefficient is the intercept
                    – Next  coefficients ( are the coefficients of the  polynomial terms for the first predictor (as specified in the polynomial_order array)
                    – Continuing in this manner for all the predictors specified in the predictor_names parameter
                    - Array length must equal the sum of the values in the polynomial_order array plus one
            interaction_term_betas:  
                A np.array-like lower triangular matrix with both dimensions equal to the sum of the 
                polynomial_order array containing the betas of any interaction terms
        
            epsilon_variance:  
                A scalar variance specification
        
        Raises:
            KeyError: If the column does not exists.

        """     
        def _is_len_match_list(list1, list1_name, list2, list2_name):
            if not len(list1) == len(list2):
                raise ValueError(f'{list1_name} and {list2_name} must have same length')
            return True
        
        def _valid_input(predictor_name_list, polynomial_order, beta, 
                        interaction_term_betas):
            cond2 = _is_len_match_list(predictor_name_list, "predictor", polynomial_order, "polynomial_order")
            # cond3 = _is_len_match_list(interaction_term_betas, "interaction beta row", interaction_term_betas, "interaction beta column")
            if len(beta) != sum(polynomial_order) + 1:
                raise ValueError(f'length of beta must equal to length of polynomial_order plus one')
            return cond2
        
        eps = epsilon_variance * np.random.randn(self.n)
        beta = np.array(beta)

        # add polynomial term            
        poly_terms = DataFrame()
        for i in range(len(predictor_name_list)):
            pred_name = predictor_name_list[i]
            for j in range(1, polynomial_order[i] + 1):
                col_name = pred_name + "^" + str(j)
                poly_terms[col_name] = self.predictor_matrix[pred_name] ** j

        # add interaction terms
        interact_terms = DataFrame()
        for c1, c2 in combinations(poly_terms.columns, 2):
            interact_terms['{0}*{1}'.format(c1,c2)] = poly_terms[c1] * poly_terms[c2]
        
        # iterate interaction term betas
        interact_betas = []
        for i in range(1, len(interaction_term_betas)):
            for j in range(i):
                interact_betas.append(interaction_term_betas[i][j])
        
        poly_mul_beta = safe_sparse_dot(poly_terms, beta[1:].T, dense_output=True)
        in_mul_beta = safe_sparse_dot(interact_terms, np.array(interact_betas).T, dense_output=True)
        self.response_vector = beta[0] + poly_mul_beta + in_mul_beta + eps
