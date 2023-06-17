<div align="center">
<br/>

[![PyPI Shield](https://img.shields.io/pypi/v/analyticsdf?color=blue)](https://pypi.org/project/analyticsdf/)
[![Unit Tests](https://github.com/Faye-yufan/analytics-dataset/actions/workflows/unit.yml/badge.svg?branch=main)](https://github.com/Faye-yufan/analytics-dataset/actions/workflows/unit.yml?query=branch%3Amain)
[![License](https://img.shields.io/github/license/Faye-yufan/analytics-dataset)](https://github.com/Faye-yufan/analytics-dataset/blob/edit-readme/LICENSE)

<div align="center">
<br/>
<p align="center">
<a href="https://github.com/Faye-yufan/analytics-dataset/">
<img align="center" width=40% src="https://github.com/Faye-yufan/analytics-dataset/blob/edit-readme/docs/images/autogen-logo.png"></img>
</a>
</p>
</div>

</div>

# Overview
The AutoGen (analyticsdf) is a Python library that allows you to generate synthetic data with any statistical characteristics desired.

## Features
This library provides a set of functionality to enable the specification and generation of a wide range of datasets with specified statistical characteristics. Specification includes the predictor matrix and the response vector.

Some common congifuration:
* High correlation and multi-collinearity among predictor variables
* Interaction effects between variables
* Skewed distributions of predictor and response variables
* Nonlinear relationships between predictor and response variables

Check the [Analyticsdf documentation](https://faye-yufan.github.io/analytics-dataset/) for more details.

## Inspirations
* Sklearn [Make Datasets](https://scikit-learn.org/stable/datasets/sample_generators.html) functionality
* MIT Synthetic Data Vault project
  * [MIT Data to AI Lab](https://dai.lids.mit.edu/)
  * [datacebo](https://datacebo.com/)
  * 2016 IEEE conference paper, The Synthetic Data Vault. 


# Install
The beta package of this library is publicly available on both [PyPI](https://pypi.org/project/analyticsdf/) and [Anaconda](https://anaconda.org/faye-yufan/analyticsdf).
Install analyticsdf using pip or conda. We recommend using a virtual environment to avoid conflicts with other software on your device.

```bash
pip install analyticsdf
```

```bash
conda install -c faye-yufan analyticsdf
```

# Getting Started
Import the dataset generation class from the package, and play with the class functions.

```python
from analyticsdf.analyticsdataframe import AnalyticsDataframe
ad = AnalyticsDataframe(1000, 6)
ad.predictor_matrix.head()
```

![Initialized Predictor Matrix](https://github.com/Faye-yufan/analytics-dataset/blob/edit-readme/docs/images/initialized-predictor-matrix.png)

The predictor matrix is initialized with all null values.
Now let's update the predictors with some distributions:

```python
for var in ['X1', 'X2', 'X3', 'X4', 'X5']:
        ad.update_predictor_uniform(var, 0, 100)
ad.update_predictor_categorical('X6', ["Red", "Yellow", "Blue"], [0.3, 0.4, 0.3])
```

![Updated Predictor Matrix](https://github.com/Faye-yufan/analytics-dataset/blob/edit-readme/docs/images/updated-predictor-matrix.png)

Once we have a dataframe desired and would like to visualize it, we can do:

```python
df_visualization_bi(ad)
```

![Bivariate Visualization Chart](https://github.com/Faye-yufan/analytics-dataset/blob/edit-readme/docs/images/bivariate-vis.png)


# Next Steps
We plan to integrate an user interface to the library, aiming to let users configure, manipulate, and view datasets more easily.


## Code Contributors
![Contributors](https://github.com/Faye-yufan/analytics-dataset/blob/edit-readme/docs/images/contributors.png)

## License
AutoGen is released under the MIT License.





