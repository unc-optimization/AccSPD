# AccSPD
An accelerated Stochastic Primal Dual Algorithm

## Introduction
This Algorithm can solve the following nonsmooth composite convex optimization problem:

<img src="https://render.githubusercontent.com/render/math?math=$\min_{x \in \mathbb{R}^p}\Big\{ f(x)$"> 

<img src="https://latex.codecogs.com/gif.latex?P(s | O_t )=\text { Probability of a sensor reading value when sleep onset is observed at a time bin } t " />


where <img src="https://render.githubusercontent.com/render/math?math=$f:\mathbb{R}^p\to\mathbb{R}\cup\{+\infty\}$"> <img src="https://render.githubusercontent.com/render/math?math=$g:\mathbb{R}^d\to\mathbb{R}\cup\{+\infty\}$"> are proper, closed and convex functions ,<img src="https://render.githubusercontent.com/render/math?math=$h:\mathbb{R}^p\to\mathbb{R}\cup\{+\infty\}$"> is a convex and smooth function, and <img src="https://render.githubusercontent.com/render/math?math=$K:\mathbb{R}^p \to\mathbb{R}^d$"> is a given linear operator. We further assume that the prox-operator of <img src="https://render.githubusercontent.com/render/math?math=$f, g$"> are easy to find.

## Prerequisites

The code is tested under Python 3.6 and it requires additional packages if you do not have them

* scipy: for working with datasets
* pickle: for saving and loading the data
* matplotlib: for plotting
* sklearn: for loading the LIBSVM data
* numpy: for scentific computing

These packages can be installed by
```
pip3 install scipy pickle matplotlib sklearn numpy
```
We support LIBSVM datasets which can be downloaded [here](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html). The downloaded dataset should be unzipped and put in the following folder

```
./data/
```

## Running the examples

we implemented two examples to test our algorithm.

### 1. Support Vector Machine (SVM) example

To run the SVM examle, use the commend below:
* mini batch case
```
python3 SVM_example.py
```
* single sample case
```
python3 SVM_example_single_sample.py
```

### 2. L1-regularized least absolute deviation (LAD) example

To run the LAD examle, use the commend below:
* mini batch case
```
python3 LAD_example.py
```

## Testing the convergence rate

To test the convergence of our algorithms for non-strongly convex case, use the commend below:
```
python3 compare_c.py
```
To test the convergence of our algorithms for strongly convex case, use the commend below:
```
python3 compare_c_str_cvx.py
```

