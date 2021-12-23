import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:

    e_x = np.exp(x)

    return e_x / e_x.sum(axis=0)
