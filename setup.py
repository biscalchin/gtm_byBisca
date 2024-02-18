from setuptools import setup, find_packages
import codecs
import os

VERSION = '1.0.0'
DESCRIPTION = 'Generative Topographic Mapping and Analysis Toolkit'
LONG_DESCRIPTION = (
    'A comprehensive toolkit for applying Generative Topographic Mapping (GTM) to '
    'high-dimensional data analysis. This package includes tools for GTM model fitting, '
    'dimensionality reduction, error analysis (k3n-error), and hyperparameter optimization '
    'via cross-validation. It supports visualization of high-dimensional datasets in lower-dimensional '
    'spaces and offers methods for both forward and inverse mappings, making it an invaluable resource '
    'for researchers and practitioners in fields such as machine learning, data science, and bioinformatics.'
)

# Setting up
setup(
    name="GTMAnalysisToolkit-byBisca",  # A name that reflects the package's purpose
    version=VERSION,
    author="Eng. Alberto Biscalchin",  # Replace with your name
    author_email="<biscalchin.mau.se@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',  # Assuming scipy is used for distance calculations and more
        'matplotlib',  # For plotting and visualization of GTM results
        'scikit-learn',  # scikit-learn for PCA, k-nearest neighbors, etc.
    ],
    keywords=[
        "generative topographic mapping", "dimensionality reduction", "data visualization",
        "machine learning", "high-dimensional data analysis"
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",  # Update as appropriate
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)

