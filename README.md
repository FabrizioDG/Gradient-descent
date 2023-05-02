## Gradient Descent implementation for multiple variable linear regression

In this repository I implement the gradient descent method, using numpy vectorization, and I compare the results of my implementation with the analytical solution of the normal equation, and with the results using the implementation of Scikit-learn library. This work is a small project I carried out during the Data Science bootcamp at the school "The Bridge" in Valencia (Spain).

### Structure of the repository

#### Folder src/

This folder contains a few functions where I implement the actual gradient descent:

- cost_function(X, y, w, b), where:
   - X (ndarray (m,n)): Data, m training examples, n features
   - y ((ndarray (m,)): Target, m examples 
   - w (ndarray (n,)): weights, n features
   - b (scalar) : bias

   Evaluate the cost function $\frac{1}{m} \sum_1^m{ (y - \hat{y})^2} $

- gradient(X, y, w, b, clip_value = None):
    - X (ndarray (m,n)): Data, m training examples, n features
    - y ((ndarray (m,)): Target, m examples 
    - w (ndarray (n,)): weights, n features
    - b (scalar) : bias

    Evaluate the gradient of the cost function w.r.t. the parameters w and b

- gradient_descent(X, y, w_in, b_in, alpha, tolerance = 1e-10, num_iter = 100000, clip_value=None):
    - X (ndarray (m,n)): Data, m training examples, n features
    - y ((ndarray (m,)): Target, m examples 
    - w_in (ndarray (n,)): initial guess for weights, n features
    - b_in (scalar) : initial guess for bias
    - alpha (scalar) : learning rate
    - tolerance (scalar) : tolerance for convergence
    - num_iter (scalar) : maximum number of iterations

    Gradient descent to minimize the cost function

The folder also contains two functions to generate a gif to show how starting from an initial guess, the code reaches the line which fit best the points.

#### Folder notebooks/

In this folder you can find the notebook where the implemented gradient descent code is used for a few examples, and where I implement the analytic solution using the normal equation, and I compare the results of these two methods.