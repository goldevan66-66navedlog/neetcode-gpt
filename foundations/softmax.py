import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)

        mz = np.max(z,axis=0)
        z = np.exp(z-mz)
        total = np.sum(z)
        return np.round(z/total,4)
