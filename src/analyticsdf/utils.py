def check_update_normal(func):
    def inner(self, predictor_name_list, mean, covariance_matrix):

        _validate_columns_exist(self.predictor_matrix, predictor_name_list)
        _is_len_matched(predictor_name_list, mean)
        _is_len_matched(mean, covariance_matrix[0])

        return func(self, predictor_name_list, mean, covariance_matrix)
    return inner

def check_update_beta(func):
    def inner(self, a, b, predictor_name_list):

        _validate_columns_exist(self.predictor_matrix, predictor_name_list)
        _is_len_matched(predictor_name_list, a, b)

        return func(self, a, b, predictor_name_list)
    return inner

def check_response_linear(func):
    def inner(self, beta, epsilon_variance, predictor_name_list):

        _validate_columns_exist(self.predictor_matrix, predictor_name_list)
        _is_len_matched(predictor_name_list, beta[1:])

        return func(self, beta, epsilon_variance, predictor_name_list)
    return inner

def _validate_columns_exist(X, names):
    actual = X.columns.values.tolist()
    missing = set(names) - set(actual)
    if missing:
        raise KeyError(f'The columns {missing} were not found in predictors.')

def _is_len_matched(*args):
    lengths = [len(arg) if isinstance(arg, list) else 1 for arg in args ]
    if lengths.count(lengths[0]) != len(lengths):
        raise Exception(f'Arguments must have the same length')
