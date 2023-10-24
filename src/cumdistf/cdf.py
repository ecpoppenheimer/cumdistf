"""
Convenience classes to handle 1 and 2 dimensional discrete cumulative distribution functions / quantile functions.
"""

import numpy as np
from scipy.interpolate import interp1d


class CumulativeDistributionFunction1D:
    """
    1D cumulative distribution function and/or quantile function.

    A cumulative distribution function is a mathematical tool that can map a uniform random sample onto some non-uniform
    distribution.  A quantile function / inverse CDF does the opposite: map a non-uniform distribution onto a uniform
    region.  These two operations can even be chained to build a mapping between two different non-uniform
    distributions.

    This class acts both computes a CDF / iCDF and also can act as store of the data used to build them.  The
    constructor requires the working domain as a parameter, and may or may not receive a density function.  If a
    density function is provided to the constructor, it will build the CDF immediately, but density can be omitted
    during class construction and added later using the accumulate functions.  Accumulate_density() accepts an already
    evaluated density function, while accumulate_points() accepts and bins a set of points to form the density.

    Please note that by "density function" I actually mean a density array.  This class only handles discrete CDFs, that
    are defined by a density array.  This density can come from a histogram, or from evaluating an analytic function on
    a grid.  This class provides a histogram convenience function, but cannot accept an analytic function.

    If density is not given to the constructor, the class will not evaluate until compute() is called.  This allows for
    data to be generated slowly over time, and avoids unnecessary computation being wasted before all data is prepared.
    If density is given to the constructor, compute() is called right away.

    The CDF is not usable until compute() has been called.  Once compute() has been called, the class can be reset by
    calling clear_density(), which deletes all data and prepares the class to be used again on a different set of data.

    By default, this class computes both a forward cumulative distribution function and an inverse CDF or quantile
    function.  If only one is needed, the parameter "direction" can be given to compute().  "forward" corresponds to
    the CDF and "backward" to the quantile / iCDF.

    Once compute() has been called, the CDF can be evaluated by calling the class, or calling self.cdf().  The
    quantile / iCDF are accessed by calling their associated functions quantile() and icdf() are aliases of each other.
    These functions accept sets of points and return mapped points with the same shape.

    See https://en.wikipedia.org/wiki/Cumulative_distribution_function for reference.

    Parameters
    ----------
    eval_limits : tuple
        This is a tuple that defines the domain of the distribution resolution.  It must be (x_start, x_end).
    density : 2d tensor-like, optional
        Defaults to None.  If non-none, compute is called on this density by the
        constructor.
    direction : str, optional
        See compute() only has effect if density is not None.
    dtype : np dtype, optional
        The dtype to use for the density function, defaults to np.float32.  Other functions infer dtype from
        their inputs.
    """
    def __init__(self, eval_limits, density=None, direction="both", dtype=np.float32):
        # declaration of class variables
        self._res = -1
        self._cdf = None
        self._icdf = None
        self._dtype = dtype

        self.x_min, self.x_max = eval_limits
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
            self._res = self._density.shape[0]
        else:
            self._density += np.array(density, dtype=np.float32)

    def clear_density(self):
        """
        Flush accumulated probability density data.  Does not affect the CDF.
        """
        self._density = None
        self._res = -1

    def histogram_points(self, points, res=None):
        """
        Use a histogram to convert a set of points into a density.

        Just a convenience wrapper for numpy's histogram function, that correctly sets the limits.

        Parameters
        ----------
        points : np array with shape (n,)
            The points to bin.
        res : tuple of ints, optional
            The resolution at which to generate the histogram.  Will throw an error if this parameter is left out
            and the resolution of the CDF has not been set.  Will be ignored if the resolution is already set.

        Returns
        -------
        histo : np array
            The generated histogram.  It will have a shape equal to the resolution of this CDF.

        """
        if self._res == -1:
            if res is None:
                raise RuntimeError(
                    "CumulativeDistributionFunction1D: resolution must be specified if it has not already been set."
                )
        else:
            res = self._res

        histo, _ = np.histogram(
            points,
            bins=res,
            range=(self.x_min, self.x_max)
        )
        return histo.astype(self._dtype)

    def accumulate_points(self, points, res=None):
        """
        Accumulate points into the density.

        Just a chaining of histogram_points() and accumulate_density()

        Parameters
        ----------
        points : np array with shape (n)
            The points to bin.
        res : int, optional
            The resolution at which to generate the histogram.  Will throw an error if this parameter is left out
            and the resolution of the CDF has not been set.  Will be ignored if the resolution is already set.

        """
        self.accumulate_density(self.histogram_points(points, res))

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
            density = np.pad(density, ((1, 0),), mode="constant", constant_values=0)
            cumsum = np.cumsum(density)

            # rescale the sums to go between 0 and 1
            cumsum /= cumsum[-1]

            # Interpolate to generate the CDF. We need new x coordinate lists with
            # one extra element, since the cumsum adds a zero at the start.
            interpolate_x = np.linspace(self.x_min, self.x_max, self._res + 1)

            # compute the forward / normal CDF
            if do_forward:
                self._cdf = interp1d(cumsum, interpolate_x, fill_value="extrapolate")

            else:
                self._cdf = None

            # compute the inverse CDF
            if do_backward:
                self._icdf = interp1d(interpolate_x, cumsum, fill_value="extrapolate")
            else:
                self._icdf = None
        else:
            raise RuntimeError(
                "CumulativeDensityFunction1D: cannot call compute before accumulating data."
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
        if self._cdf is not None:
            return self._cdf(points).astype(points.dtype)
        else:
            raise ComputeRequiredError()

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
        if self._cdf is not None:
            return self._icdf(points).astype(points.dtype)
        else:
            raise ComputeRequiredError()

    def quantile(self, points):
        """Alias for icdf()."""
        return self.icdf(points)

    def __call__(self, points):
        """Second, default calling convention.  Calls cdf()"""
        return self.cdf(points)

    @property
    def res(self):
        return self._res


class CumulativeDistributionFunction2D:
    """
    2D cumulative distribution function and/or quantile function.

    A cumulative distribution function is a mathematical tool that can map a uniform random sample onto some non-uniform
    distribution.  A quantile function / inverse CDF does the opposite: map a non-uniform distribution onto a uniform
    region.  These two operations can even be chained to build a mapping between two different non-uniform
    distributions.

    This class both computes a CDF / iCDF and also can act as store of the data used to build them.  The
    constructor requires the working domain as a parameter, and may or may not receive a density function.  If a
    density function is provided to the constructor, it will build the CDF immediately, but density can be omitted
    during class construction and added later using the accumulate functions.  Accumulate_density() accepts an already
    evaluated density function, while accumulate_points() accepts and bins a set of points to form the density.

    Please note that by "density function" I actually mean a density array.  This class only handles discrete CDFs, that
    are defined by a density array.  This density can come from a histogram, or from evaluating an analytic function on
    a grid.  This class provides a histogram convenience function, but cannot accept an analytic function.

    If density is not given to the constructor, the class will not evaluate until compute() is called.  This allows for
    data to be generated slowly over time, and avoids unnecessary computation being wasted before all data is prepared.
    If density is given to the constructor, compute() is called right away.

    The CDF is not usable until compute() has been called.  Once compute() has been called, the class can be reset by
    calling clear_density(), which deletes all data and prepares the class to be used again on a different set of data.

    By default, this class computes both a forward cumulative distribution function and an inverse CDF or quantile
    function.  If only one is needed, the parameter "direction" can be given to compute().  "forward" corresponds to
    the CDF and "backward" to the quantile / iCDF.

    Once compute() has been called, the CDF can be evaluated by calling the class, or calling self.cdf().  The
    quantile / iCDF are accessed by calling their associated functions quantile() and icdf() are aliases of each other.
    These functions accept sets of points and return mapped points with the same shape.

    See https://en.wikipedia.org/wiki/Cumulative_distribution_function for reference.

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
    dtype : np dtype, optional
        The dtype to use for the density function, defaults to np.float32.  Other functions infer dtype from
        their inputs.
    """
    def __init__(self, eval_limits, density=None, direction="both", dtype=np.float32):
        # declaration of class variables
        self._x_res = -1
        self._y_res = -1
        self._y_cdf = None
        self._x_cdfs = None
        self._y_icdf = None
        self._x_icdfs = None
        self._dtype = dtype

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
            self._density = np.array(density, dtype=self._dtype)
            self._x_res, self._y_res = self._density.shape
        else:
            self._density += np.array(density, dtype=self._dtype)

    def histogram_points(self, points, res=None):
        """
        Use a histogram to convert a set of points into a density.

        Just a convenience wrapper for numpy's histogram function, that correctly sets the limits.

        Parameters
        ----------
        points : np array with shape (n, 2)
            The points to bin.
        res : 2-tuple of ints, optional
            The resolution at which to generate the histogram.  Will throw an error if this parameter is left out
            and the resolution of the CDF has not been set.  Will be ignored if the resolution is already set.

        Returns
        -------
        histo : np array
            The generated histogram.  It will have a shape equal to the resolution of this CDF.

        """
        if self._x_res == -1 or self._y_res == -1:
            if res is None:
                raise RuntimeError(
                    "CumulativeDistributionFunction2D: resolution must be specified if it has not already been set."
                )
            else:
                x_res, y_res = res
        else:
            x_res, y_res = self._x_res, self._y_res

        histo, _, _ = np.histogram2d(
            points[:, 0],
            points[:, 1],
            bins=(x_res, y_res),
            range=((self.x_min, self.x_max), (self.y_min, self.y_max))
        )
        return histo.astype(self._dtype)

    def accumulate_points(self, points, res=None):
        """
        Accumulate points into the density.

        Just a chaining of histogram_points() and accumulate_density()

        Parameters
        ----------
        points : np array with shape (n, 2)
            The points to bin.
        res : 2-tuple of ints, optional
            The resolution at which to generate the histogram.  Will throw an error if this parameter is left out
            and the resolution of the CDF has not been set.  Will be ignored if the resolution is already set.

        """
        self.accumulate_density(self.histogram_points(points, res))

    def clear_density(self):
        """
        Flush accumulated probability density data.  Does not affect the CDF.
        """
        self._density = None
        self._x_res = -1
        self._y_res = -1

    def set_resolution(self, x_res, y_res):
        if self._density is not None:
            raise RuntimeError("CumulativeDistributionFunction2D: resolution can only be set if density is cleared.")
        else:
            self._x_res, self._y_res = x_res, y_res

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

            interpolate_x = np.linspace(self.x_min, self.x_max, self._x_res + 1)
            interpolate_y = np.linspace(self.y_min, self.y_max, self._y_res + 1)

            # compute the forward / normal CDF
            if do_forward:
                self._y_cdf = interp1d(y_sum, interpolate_y, fill_value="extrapolate")
                self._x_cdfs = [
                    interp1d(x_sums[:, i], interpolate_x, fill_value="extrapolate")
                    for i in range(self._y_res)
                ]
            else:
                self._y_cdf = None
                self._x_cdfs = None

            # compute the inverse CDF
            if do_backward:
                self._y_icdf = interp1d(interpolate_y, y_sum, fill_value="extrapolate")
                self._x_icdfs = [
                    interp1d(interpolate_x, x_sums[:, i], fill_value="extrapolate")
                    for i in range(self._y_res)
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
            x_curve = (y_out - self.y_min) * self._y_res / (self.y_max - self.y_min)
            x_curve = np.floor(x_curve).astype("int")
            # Ensure that x_curve is always < self._y_res (could be equal if y_out = y_max).
            x_curve = np.where(x_curve < self._y_res, x_curve, self._y_res - 1)

            # map the x coordinate.
            x_range = np.arange(x.shape[0])
            x_out = np.zeros_like(x)
            for i in range(self._y_res):
                mask = x_curve == i
                x_out[x_range[mask]] = self._x_cdfs[i](x[mask])

            x_out = x_out.astype(points.dtype)
            y_out = y_out.astype(points.dtype)
            return np.column_stack((x_out, y_out))
        else:
            raise ComputeRequiredError()

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
            x_curve = y_out * self._y_res
            x_curve = np.floor(x_curve).astype("int")
            # Ensure that x_curve is always < self._y_res (could be equal if y_out = 1).
            x_curve = np.where(x_curve < self._y_res, x_curve, self._y_res-1)

            # map the x coordinate.
            x_range = np.arange(x.shape[0])
            x_out = np.zeros_like(x)
            for i in range(self._y_res):
                mask = x_curve == i
                x_out[x_range[mask]] = self._x_icdfs[i](x[mask])

            x_out = x_out.astype(points.dtype)
            y_out = y_out.astype(points.dtype)
            return np.column_stack((x_out, y_out))
        else:
            raise ComputeRequiredError()

    def quantile(self, points):
        """Alias for icdf()."""
        return self.icdf(points)

    def __call__(self, points):
        """Second, default calling convention.  Calls cdf()"""
        return self.cdf(points)

    @property
    def x_res(self):
        return self._x_res

    @property
    def y_res(self):
        return self._y_res

    @property
    def dtype(self):
        return self._dtype


class ComputeRequiredError(RuntimeError):
    def __str__(self):
        return "CumulativeDensityFunction: Must call compute() with the correct direction before evaluation."
