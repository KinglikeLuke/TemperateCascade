from numbers import Real
import numpy as np

"""
Here all global functions are stored - the functions are up to "choice": Here, linear functions are used
"""


class global_functions():
    """
    Linear feedback function to compute feedbacks. Maximal feedback is obtained from 2.0°C onwards.
    feedbacks for Arctic summer sea ice, mountain glaciers, GIS and WAIS are proven to be constant also for higher temperatures
    feedbacks for Amazon and feedbacks_steffen are computed for a temperature increase of 2.0°C until 2100, see paper
    """
    @staticmethod
    def feedback_function(GMT, fbmax):
        # fbmax = maximal feedback
        # only returns a value in case GMT is higher than lower boundary of the respective tipping element, otherwise return 0.0 (lower cap), N.B.: No upper cap
        if GMT <= 2.0:
            y = (fbmax / 2.0) * GMT
            return y
        elif GMT > 2.0:
            return fbmax
        else:
            raise Exception("GMT negativ: Feedbacks do not work for temperatures smaller 0!")

    # feedbacks of state dependent variables, linear increase of feedbacks between state -1 and +1 for GIS, WAIS, THC, NINO and AMAZ
    @staticmethod
    def state_feedback(state, fbmax):
        if state >= -1 and state <= 1:
            y = fbmax / 2 * (state + 1)
        elif state < -1:
            y = 0.
        elif state > +1:
            y = fbmax
        return y

    # Dont worry about this mess of an architecture...
    class CUSP:
        def __init__(self, tau, x1, x2, x, y2=np.sqrt(4 / 27)):
            self.x1 = x1
            self.x2 = x2
            self.x = x
            self.y2 = y2
            self.tau = tau

        @property
        def x(self):
            return self._x

        @x.setter
        def x(self, x):
            if isinstance(x, Real):
                x = lambda t, xt=x: xt
            self._x = x

        def __call__(self, t):
            return (1.0 / self.tau) * global_functions.CUSPc(self.x1, self.x2, self.x(t), self.y2)

    # c = c(GMT) where tipping occurs at sqrt(4/27) ~ 0.38
    # Linear function through two points maps GMT --> c, where x-values represent GMT and y-values represent CUSP-c values
    @staticmethod
    def CUSPc(x1, x2, x, y2=np.sqrt(4 / 27)):
        """
        Computes the output value of a cusp-shaped function
        y/(x2-x1) *(x-x1),
        returning a value scaled
        between a lower boundary and a specified upper boundary whenever the input `x` is
        beyond or equal to the threshold `x1`. If the input `x` is below the threshold `x1`,
        it returns a fixed lower cap value of 0.0.

        This function scales linearly between 0.0 and a maximum value
        (calculated as the square root of 4/27), based on the relative
        position of `x` between `x1` and `x2`.

        Args:
            x1 (float): The lower boundary (threshold) for the input `x`.
            x2 (float): The upper boundary for linearly scaling the output value.
            x (float): The input value to evaluate against the boundaries.
            y2 (float, optional): Scaling for fold locations. Defaults to cusp
        Returns:
            float: A value scaled between 0.0 and a computed upper boundary if
            `x` >= `x1`. Returns 0.0 if `x` < `x1`.
        """
        # only returns a value in case GMT is higher than lower boundary of the respective tipping element, otherwise return 0.0 (lower cap), N.B.: No upper cap

        if x >= x1:
            y1 = 0.0
            y = (y2 - y1) / (x2 - x1) * (x - x1) + y1 # y2*T/T_thresh
            return y
        else:
            return 0.0
