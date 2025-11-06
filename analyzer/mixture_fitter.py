import numpy as np

class BinaryMixtureFitter:
    def __init__(self, fitter, theta, q_grid=None, qmin=0.1, qmax=1.0):
        """
        fitter: your MISTFitter instance (already fit)
        theta:  (age, feh, distance, AV, dM, dC) from median posterior
        """
        self.fitter = fitter
        self.theta = theta
        self.q_grid = q_grid or np.round(np.linspace(qmin, qmax, 10), 2)
