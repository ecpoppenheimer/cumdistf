import unittest

import numpy as np

import cumdistf


class UnitTester(unittest.TestCase):
    def assert_close(self, data, target, epsilon):
        self.assertLess(abs(data - target), epsilon)


class TestCase1D(UnitTester):
    def test_range(self):
        res = 20
        x_lims = (-2.0, 2.0)
        sample_count = 10000
        epsilon = .01

        # build the cdf
        x = np.linspace(*x_lims, res)
        density = np.exp(-x**2) + 1
        cdf = cumdistf.CumulativeDistributionFunction1D(x_lims, density=density)

        # test that the range produced by the cdf is equal to the x_lims
        random_sample = np.random.uniform(0.0, 1.0, (sample_count,))
        outp = cdf(random_sample)
        min_out, max_out = np.amin(outp), np.amax(outp)
        self.assert_close(min_out, x_lims[0], epsilon)
        self.assert_close(max_out, x_lims[1], epsilon)

        # test that the range produced by the icdf is equal to (0, 1)
        random_sample = np.random.uniform(*x_lims, (sample_count,))
        outp = cdf.icdf(random_sample)
        min_out, max_out = np.amin(outp), np.amax(outp)
        self.assert_close(min_out, 0.0, epsilon)
        self.assert_close(max_out, 1.0, epsilon)

    def test_point_accumulation(self):
        res = 20
        x_lims = (-2.0, 2.0)
        sample_count = 100
        epsilon = .01

        # build the cdf
        cdf = cumdistf.CumulativeDistributionFunction1D(x_lims)
        # Can't call compute until density has been added
        self.assertRaises(RuntimeError, cdf.compute)
        # Can't accumulate because there is no density
        self.assertRaises(RuntimeError, cdf.accumulate_points, np.random.normal(1.0, .5, (sample_count,)))
        # Can accumulate if res is specified
        cdf.accumulate_points(np.random.normal(1.0, .5, (sample_count,)), res=res)


class TestCase2D(UnitTester):
    def test_range(self):
        self.range_sub((20, 20), ((-2.0, 2.0), (-3.0, 3.0)))
        self.range_sub((20, 30), ((-1.0, 2.0), (-0.0, 3.0)))
        self.range_sub((30, 20), ((0.0, 2.0), (-1.0, 1.0)))

    def range_sub(self, res, lims):
        x_res, y_res = res
        x_lims, y_lims = lims
        sample_count = 10000
        epsilon = .01

        # build the cdf
        x = np.linspace(*x_lims, x_res)[:, None]
        y = np.linspace(*y_lims, y_res)[None, :]
        density = np.exp(-(x**2 + y**2)) + 1
        cdf = cumdistf.CumulativeDistributionFunction2D(lims, density=density)

        # test that the range produced by the cdf is equal to the x_lims
        random_sample = np.random.uniform(0.0, 1.0, (sample_count, 2))
        outp = cdf(random_sample)
        min_x, max_x = np.amin(outp[:, 0]), np.amax(outp[:, 0])
        min_y, max_y = np.amin(outp[:, 1]), np.amax(outp[:, 1])
        self.assert_close(min_x, x_lims[0], epsilon)
        self.assert_close(max_x, x_lims[1], epsilon)
        self.assert_close(min_y, y_lims[0], epsilon)
        self.assert_close(max_y, y_lims[1], epsilon)

        # test that the range produced by the icdf is equal to (0, 1)
        random_x = np.random.uniform(*x_lims, (sample_count,))
        random_y = np.random.uniform(*y_lims, (sample_count,))
        random_sample = np.stack((random_x, random_y), axis=1)
        outp = cdf.icdf(random_sample)
        min_x, max_x = np.amin(outp[:, 0]), np.amax(outp[:, 0])
        min_y, max_y = np.amin(outp[:, 1]), np.amax(outp[:, 1])
        self.assert_close(min_x, 0.0, epsilon)
        self.assert_close(max_x, 1.0, epsilon)
        self.assert_close(min_y, 0.0, epsilon)
        self.assert_close(max_y, 1.0, epsilon)

        # check that the cdf generated the correct number of x_cdfs
        self.assertEqual(len(cdf._x_cdfs), y_res)
        self.assertEqual(len(cdf._x_icdfs), y_res)

    def test_point_accumulation(self):
        res = 20
        x_lims = (-2.0, 2.0)
        sample_count = 100
        epsilon = .01

        # build the cdf
        cdf = cumdistf.CumulativeDistributionFunction1D(x_lims)
        # Can't call compute until density has been added
        self.assertRaises(RuntimeError, cdf.compute)
        # Can't accumulate because there is no density
        self.assertRaises(RuntimeError, cdf.accumulate_points, np.random.normal(1.0, .5, (sample_count,)))
        # Can accumulate if res is specified
        cdf.accumulate_points(np.random.normal(1.0, .5, (sample_count,)), res=res)


if __name__ == '__main__':
    unittest.main()
