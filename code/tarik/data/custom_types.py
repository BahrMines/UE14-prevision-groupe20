from collections.abc import Callable

import numpy as np
import pandas as pd

type data_series = pd.Series | np.ndarray
type activation = Callable[[np.ndarray], np.ndarray]
