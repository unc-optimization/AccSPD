# AccSPD
An accelerated Stochastic Primal Dual Algorithm

## Introduction
This Algorithm can solve the following nonsmooth composite convex optimization problem:

<img src="http://www.sciweavers.org/tex2img.php?eq=%20%5Ccenter%20%5Cmin_%7Bx%20%5Cin%20%5Cmathbb%7BR%7D%5Ep%7D%5CBig%5C%7B%20f%28x%29%20%2B%20h%28x%29%20%2B%20g%28Kx%29%5CBig%5C%7D&bc=White&fc=Black&im=jpg&fs=18&ff=iwona&edit=0" align="center" border="0" alt=" \min_{x \in \mathbb{R}^p}\Big\{ f(x) + h(x) + g(Kx)\Big\}" width="348" height="48" />

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

