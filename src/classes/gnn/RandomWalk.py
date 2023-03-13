
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu,True)

import numpy as np
import warnings
from collections import defaultdict, deque
from scipy import stats
from scipy.special import softmax
from tqdm import tqdm

from stellargraph.core.utils import is_real_iterable
from stellargraph.core.validation import require_integer_in_range
from stellargraph.random import random_state
from abc import ABC, abstractmethod
from stellargraph import StellarGraph

def _default_if_none(value, default, name, ensure_not_none=True):
    value = value if value is not None else default
    if ensure_not_none and value is None:
        raise ValueError(
            f"{name}: expected a value to be specified in either `__init__` or `run`, found None in both"
        )
    return value

class RandomWalk(ABC):
    """
    Abstract base class for Random Walk classes. A Random Walk class must implement a ``run`` method
    which takes an iterable of node IDs and returns a list of walks. Each walk is a list of node IDs
    that contains the starting node as its first element.
    """

    def __init__(self, graph, seed=None):
        if not isinstance(graph, StellarGraph):
            raise TypeError("Graph must be a StellarGraph or StellarDiGraph.")

        self.graph = graph
        self._random_state, self._np_random_state = random_state(seed)

    def _get_random_state(self, seed):
        """
        Args:
            seed: The optional seed value for a given run.

        Returns:
            The random state as determined by the seed.
        """
        if seed is None:
            # Restore the random state
            return self._random_state, self._np_random_state
        # seed the random number generator
        require_integer_in_range(seed, "seed", min_val=0)
        return random_state(seed)

    @staticmethod
    def _validate_walk_params(nodes, n, length):
        if not is_real_iterable(nodes):
            raise ValueError(f"nodes: expected an iterable, found: {nodes}")
        if len(nodes) == 0:
            warnings.warn(
                "No root node IDs given. An empty list will be returned as a result.",
                RuntimeWarning,
                stacklevel=3,
            )

        require_integer_in_range(n, "n", min_val=1)
        require_integer_in_range(length, "length", min_val=1)

    @abstractmethod
    def run(self, nodes, **kwargs):
        pass