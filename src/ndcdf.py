"""
N-d cumulative distribution function / quantile function.
"""

import numpy as np
from scipy.interpolate import interp1d


class CumulativeDistributionFunction:
    """
    Class based implementation of the above flatten distribution function, since I have
    realised that I will in fact often want to perform the CDF operation on a different
    set of data than was used to generate it (need higher resolution on the CDF generation
    than in the utilization).  This class can compute both a cumulative density function
    (forward direction, can be used to map a uniform distribution into a non-uniform
    one) and an inverse cumulative density function (backward direction, can be used to
    map a non-uniform distribution into a uniform one).

    Note that this function uses numpy / scipy, and not tensorflow.

    Parameters
    ----------
    eval_limits : tuple
        This is a nested tuple that defines the domain of the distribution and its sampling
        resolution.  It must be ((x_start, x_end), (y_start, y_end)).
    density : 2d tensor-like, optional
        Defaults to None.  If non-none, compute is called on this density by the
        constructor.
    direction : str, optional
        See compute() only has effect if density is not None.
    """
    def __init__(self, eval_limits, density=None, direction="both"):
        # declaration of class variables
        self.x_res = 10
        self.y_res = 10
        self._y_cdf = None
        self._x_cdfs = None
        self._y_icdf = None
        self._x_icdfs = None

        self.x_min, self.x_max = eval_limits[0]
        self.y_min, self.y_max = eval_limits[1]
        if density is None:
            self._density = None
        else:
            self.compute(density, direction)

    def accumulate_density(self, density):
        """
        Add histogramed density data to the storage, but DOES NOT UPDATE THE CDF!
        Parameters
        ----------
        density : array_like
            A batch of 2D array of probability density data (typically will come
            from a histogram).  Must be the same shape and dtype as previously
            submitted data.

        """
        if self._density is None:
            self._density = np.array(density, dtype=np.float32)
            self.x_res, self.y_res = self._density.shape
        else:
            self._density += np.array(density, dtype=np.float32)

    def clear_density(self):
        """
        Flush accumulated probability density data.  Does not affect the CDF.
        """
        self._density = None

    def compute(self, density=None, direction="both", epsilon=1e-10):
        """

        Parameters
        ----------
        density : array_like, optional
            If None, defaults to use the density that has been accumulated by previous
            calls to accumulate_density, if you desire to use that calling convention.
            If non-None, this data is used instead and ACCUMULATED DATA IS CLEARED.

            A batch of 2D array of probability density data (typically will come
            from a histogram).

        direction : string, optional
            Defaults to "both".  May be "forward", "inverse" or "both".  Chooses which
            kind of CFD to compute, where forward is the standard.

        epsilon : float, optional
            Defaults to 1e-6.  A small value added everywhere to the density to
            prevent divide by zero, which can happen when an entire row has zero density.

        """
        # set the density, if it was specified
        if density is not None:
            self.clear_density()
            self.accumulate_density(density)

        # process the direction option
        if direction not in {"forward", "inverse", "both"}:
            raise ValueError(
                "CumulativeDensityFunction: direction must be one of {'forward', "
                "'backward', 'both'}"
            )
        do_forward = False
        do_backward = False
        if direction in {"forward", "both"}:
            do_forward = True
        if direction in {"inverse", "both"}:
            do_backward = True

        # Compute the cumulative density
        if self._density is not None:
            # pad the density function, since we want our cumsums to start from zero.
            density = self._density + epsilon
            density = np.pad(density, ((1, 0), (1, 0)), mode="constant", constant_values=0)
            x_sums = np.cumsum(density, axis=0)
            y_sum = np.cumsum(x_sums[-1])  # sum along the last element to get everything
            x_sums = x_sums[:, 1:]  # now we can remove the pad column

            # y sum is the cumulative sum along the y dimension, independent of x
            # i.e. all x data is compressed into a single bin.
            # x sums are the cumulative sums along the x dimension, for each segment
            # of y values

            # rescale the sums to go between 0 and 1
            y_sum /= y_sum[-1]
            x_sums /= x_sums[-1:]

            # Interpolate to generate the CDF. We need new x and y coordinate lists with
            # one extra element, since the cumsum adds a zero at the start.

            interpolate_x = np.linspace(self.x_min, self.x_max, self.x_res + 1)
            interpolate_y = np.linspace(self.y_min, self.y_max, self.y_res + 1)

            # compute the forward / normal CDF
            if do_forward:
                self._y_cdf = interp1d(y_sum, interpolate_y)
                self._x_cdfs = [
                    interp1d(x_sums[:, i], interpolate_x)
                    for i in range(self.y_res)
                ]
            else:
                self._y_cdf = None
                self._x_cdfs = None

            # compute the inverse CDF
            if do_backward:
                self._y_icdf = interp1d(interpolate_y, y_sum)
                self._x_icdfs = [
                    interp1d(interpolate_x, x_sums[:, i])
                    for i in range(self.y_res)
                ]
            else:
                self._x_icdfs = None
                self._y_icdf = None
        else:
            raise RuntimeError(
                "CumulativeDensityFunction: cannot call compute before "
                "accumulating data."
            )

    @staticmethod
    def _rescale(n, n_min, n_max):
        return n * (n_max - n_min) / np.amax(n) + n_min

    def cdf(self, points):
        """
        Evaluate the cumulative density function on a set of points.

        Compute must be called first, to compute the cumulative sums, this
        function only evaluates given a set of points.

        Please note that this function maps input points from the domain (0, 1)
        onto the output domain defined by the eval limits specified to the
        constructor.

        Parameters
        ----------
        points : tensor-like with shape (n, 2)
            The points to map, using the (forward) cumulative density function.
            If points are uniformly distributed, this function will map them so that
            their density matches the density of this CDF.

        Returns
        -------
        output : tensor-like with same shape and dtype as points
            The mapped points.
        """
        if self._y_cdf is not None:
            x = points[:, 0]
            y = points[:, 1]

            # map the y coordinate first.
            y_out = self._y_cdf(y)

            # select which x quantile curve to use.
            x_curve = (y_out - self.y_min) * self.y_res / (self.y_max - self.y_min)
            x_curve = np.floor(x_curve).astype("int")

            # map the x coordinate.
            x_range = np.arange(x.shape[0])
            x_out = np.zeros_like(x)
            for i in range(self.y_res):
                mask = x_curve == i
                x_out[x_range[mask]] = self._x_cdfs[i](x[mask])

            x_out = x_out.astype(points.dtype)
            y_out = y_out.astype(points.dtype)
            return np.column_stack((x_out, y_out))
        else:
            raise RuntimeError(
                "CumulativeDensityFunction: Must call compute() with the correct "
                "direction before evaluation."
            )

    def icdf(self, points):
        """
        Evaluate the inverse cumulative density function on a set of points.

        Compute must be called first, to compute the cumulative sums, this
        function only evaluates given a set of points.

        Please note that this function maps input points from the domain
        defined by the eval limits specified to the constructor onto the output
        domain (0, 1).

        Parameters
        ----------
        points : tensor-like with shape (n, 2)
            The points to map, using the (backward) inverse cumulative density function.
            If points are distributed like the density of this CDF, this function will
            map them onto a uniform distribution.

        Returns
        -------
        output : tensor-like with same shape and dtype as points
            The mapped points.
        """
        if self._y_icdf is not None:
            x = points[:, 0]
            y = points[:, 1]

            # map the y coordinate first.
            y_out = self._y_icdf(y)

            # select which x quantile curve to use.
            x_curve = y_out * (self.y_res - 1)
            x_curve = np.floor(x_curve).astype("int")

            # map the x coordinate.
            x_range = np.arange(x.shape[0])
            x_out = np.zeros_like(x)
            for i in range(self.y_res):
                mask = x_curve == i
                x_out[x_range[mask]] = self._x_icdfs[i](x[mask])

            x_out = x_out.astype(points.dtype)
            y_out = y_out.astype(points.dtype)
            return np.column_stack((x_out, y_out))
        else:
            raise RuntimeError(
                "CumulativeDensityFunction: Must call compute() with the correct "
                "direction before evaluation."
            )

    def quantile(self, points):
        """Alias for icdf()."""
        return self.icdf(points)

    def __call__(self, points):
        """Second, default calling convention.  Calls cdf()"""
        return self.cdf(points)
