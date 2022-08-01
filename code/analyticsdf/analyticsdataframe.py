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
