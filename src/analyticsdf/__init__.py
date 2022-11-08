"""Top-level package for analyticsdataframe."""

import numpy as np
import pandas as pd

def check_columns_exist(function):
    """Raise an exception if the given columns does not exists in dataframe.

    Args:
        function(callable): Method whose arguments are a pandas.dataframe-like object and a list of column names.

    Returns:
        callable: Decorated function

    Raises:
        KeyError: If the column does not exists

    """

    def decorated(self, predictor_name_list, *args, **kwargs):
        actual = self.predictor_matrix.columns.values.tolist()

        missing = set(predictor_name_list) - set(actual)
        
        if missing:
            raise KeyError(f'The columns {missing} were not found in predictors.')
        return function(self, predictor_name_list, *args, **kwargs)
    return decorated