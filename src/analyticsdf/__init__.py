"""Top-level package for analyticsdataframe."""

import numpy as np
import pandas as pd
import contextlib
import random
from functools import wraps

@contextlib.contextmanager
def set_random_state(random_state):
    """Context manager for managing the random state.

    Args:
        random_state (int or np.random.RandomState):
            The random seed or RandomState.

    """
    original_state = np.random.get_state()

    np.random.set_state(random_state.get_state())

    try:
        yield
    finally:
        np.random.set_state(original_state)


def validate_random_state(random_state):
    """Validate random state argument.

    Args:
        random_state (int, numpy.random.RandomState, tuple, or None):
            Seed or RandomState for the random generator.

    Return:
        numpy.random.RandomState

    """
    if random_state is None:
        return np.random.RandomState(seed=random.randrange(9999))

    if isinstance(random_state, int):
        return np.random.RandomState(seed=random_state)
    elif isinstance(random_state, np.random.RandomState):
        return random_state
    else:
        raise TypeError(
            f'`random_state` {random_state} expected to be an int '
            'or `np.random.RandomState` object.')
            

def check_columns_exist(function):
    """Raise an exception if the given columns does not exists in dataframe.

    Args:
        function(callable): Method whose arguments are a pandas.dataframe-like object and a list of column names.

    Returns:
        callable: Decorated function

    Raises:
        KeyError: If the column does not exists

    """
    @wraps(function)
    def decorated(self, predictor_name_list, *args, **kwargs):
        actual = self.predictor_matrix.columns.values.tolist()
        if isinstance(predictor_name_list, list):
            missing = set(predictor_name_list) - set(actual)
        else:
            missing = set([predictor_name_list]) - set(actual)
        
        if missing:
            raise KeyError(f'The columns {missing} were not found in predictors.')
        return function(self, predictor_name_list, *args, **kwargs)
    return decorated


def check_is_numeric(col):
    return np.issubdtype(col.dtype, np.number)