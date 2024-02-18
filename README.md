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
pip install GTMAnalysisToolkit-byBisca
```
Ensure you have Python 3.x installed on your system. This package depends on numpy, scipy, matplotlib, and sklearn, which will be automatically installed during the GTMAnalysisToolkit installation.

## Quick Start

Here's a quick example to get you started with using the GTMAnalysisToolkit:

```Python
from GTMAnalysisToolkit import GTM
import numpy as np

# Example dataset
X = np.random.rand(100, 5)  # 100 samples, 5-dimensional

# Initialize GTM model
gtm_model = GTM()

# Fit model
gtm_model.fit(X)

# Transform data to lower-dimensional space
X_transformed = gtm_model.transform(X)

# Visualize the result
gtm_model.visualize(X_transformed)
```

## Documentation

WORK IN PROGRESS

## License

This project is licensed under the GNU General Public License version 3 (GPL-3.0) - see the LICENSE file for details.

## Contributing
We welcome contributions to the GTMAnalysisToolkit! If you have suggestions or want to contribute code, please feel free to open an issue or pull request on our GitHub repository.

## Contact
For questions or support, please contact Eng. Alberto Biscalchin at biscalchin.mau.se@gmail.com