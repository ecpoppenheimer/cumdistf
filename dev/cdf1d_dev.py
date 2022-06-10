import numpy as np
import matplotlib.pyplot as plt

import cumdistf

# parameters
x_range = (-3.0, 5.0)
x_res = 100
sample_count = 10000

# define an evaluation grid, and define the density
x = np.linspace(x_range[0], x_range[1], x_res)
#density = np.exp(-(x**2/4 + y**2))
density = np.exp(-x**2) + .3 * x + 1

# set up the plot.  Plot the density in the first panel
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].plot(density)
axes[0].set_title("analytic density")

# make the cdf
cdf = cumdistf.CumulativeDistributionFunction1D(x_range, density)

# make a random sample of points, which live in the domain (0, 0) -> (1, 1)
random_sample = np.random.uniform(0.0, 1.0, (sample_count,))

# evaluate the cdf on the random sample, and plot into the second pane
mapped_sample = cdf(random_sample)
histo, bin_edges = np.histogram(mapped_sample, bins=15, range=x_range)
axes[1].bar((bin_edges[1:] + bin_edges[:-1])/2, histo)
axes[1].set_title("binned density of mapped sample")

# evaluate the inverse cdf on the mapped sample, and plot into the third pane
flattened_sample = cdf.icdf(mapped_sample)
histo, bin_edges = np.histogram(flattened_sample, bins=15, range=(0, 1))
axes[2].bar((bin_edges[1:] + bin_edges[:-1])/2, histo)
axes[2].set_title("binned density of flattened mapped sample")

plt.show()
