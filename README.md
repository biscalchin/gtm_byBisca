# Generative Topographic Mapping and Analysis Toolkit
## Introduction

The Generative Topographic Mapping and Analysis Toolkit is a Python package designed for high-dimensional data analysis using Generative Topographic Mapping (GTM). This toolkit facilitates the visualization of high-dimensional datasets in lower-dimensional spaces, supports both forward and inverse mappings, and offers comprehensive tools for model fitting, dimensionality reduction, error analysis, and hyperparameter optimization. It is particularly useful for researchers and practitioners in machine learning, data science, and bioinformatics.

## Features

- GTM model fitting for dimensionality reduction and data visualization.
- k-nearest neighbor normalized error (k3n-error) calculation for error analysis.
- Cross-validation based hyperparameter optimization for GTM models.
- Support for both forward analysis (regression) and inverse analysis.
- Visualization tools for low-dimensional embeddings of high-dimensional data.

## Installation 

To install the GTMAnalysisToolkit, run the following command in your terminal:
```bash
pip install GTMAnalysisToolkit
```
Ensure you have Python 3.x installed on your system. This package depends on numpy, scipy, matplotlib, and sklearn, which will be automatically installed during the GTMAnalysisToolkit installation.

## Quick Start

Here's a quick example to get you started with using the GTMAnalysisToolkit:

```Python
import matplotlib.figure as figure
import matplotlib.pyplot as plt
from GTMAnalysisToolkit import GTM
from sklearn.datasets import load_iris

# settings
shape_of_map = [10, 10]
shape_of_rbf_centers = [5, 5]
variance_of_rbfs = 4
lambda_in_em_algorithm = 0.001
number_of_iterations = 300
display_flag = 1

# load an iris dataset
iris = load_iris()
# input_dataset = pd.DataFrame(iris.data, columns=iris.feature_names)
input_dataset = iris.data
color = iris.target

# autoscaling
input_dataset = (input_dataset - input_dataset.mean(axis=0)) / input_dataset.std(axis=0, ddof=1)

# construct GTM model
model = GTM(shape_of_map, shape_of_rbf_centers, variance_of_rbfs, lambda_in_em_algorithm, number_of_iterations,
            display_flag)
model.fit(input_dataset)

if model.success_flag:
    # calculate of responsibilities
    responsibilities = model.responsibility(input_dataset)

    # plot the mean of responsibilities
    means = responsibilities.dot(model.map_grids)
    plt.figure(figsize=figure.figaspect(1))
    plt.scatter(means[:, 0], means[:, 1], c=color)
    plt.ylim(-1.1, 1.1)
    plt.xlim(-1.1, 1.1)
    plt.xlabel("z1 (mean)")
    plt.ylabel("z2 (mean)")
    plt.show()

    # plot the mode of responsibilities
    modes = model.map_grids[responsibilities.argmax(axis=1), :]
    plt.figure(figsize=figure.figaspect(1))
    plt.scatter(modes[:, 0], modes[:, 1], c=color)
    plt.ylim(-1.1, 1.1)
    plt.xlim(-1.1, 1.1)
    plt.xlabel("z1 (mode)")
    plt.ylabel("z2 (mode)")
    plt.show()

```

## License

This project is licensed under the GNU General Public License version 3 (GPL-3.0) - see the LICENSE file for details.

## Contributing
We welcome contributions to the GTMAnalysisToolkit! If you have suggestions or want to contribute code, please feel free to open an issue or pull request on our GitHub repository.

## Contact
For questions or support, please contact Eng. Alberto Biscalchin at biscalchin.mau.se@gmail.com