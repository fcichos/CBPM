import numpy as np
from scipy.special import comb


def Dn(f, n, x, *params, h=1e-5):
    """
    Calculate the nth order derivative of function f at positions x using
    the direct formula:
    
    f^(n)(x) = lim_{h→0} (1/h^n) * sum_{k=0}^n [(-1)^(k+n) * binomial(n,k) * f(x+kh)]
    
    Parameters:
    ----------
    f : callable
        The function to differentiate. Should take x as first argument
        and any additional parameters afterward.
    n : int
        The order of the derivative to calculate.
    x : float or array_like
        Points at which to evaluate the derivative.
    *params : tuple
        Additional parameters to pass to the function f.
    h : float, optional
        Step size for differentiation (default: 1e-5).
    
    Returns:
    -------
    ndarray or float
        The nth derivative of f evaluated at positions x.
    """
    # Input validation
    if not callable(f):
        raise TypeError("f must be a callable function")
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")
    
    # Convert x to numpy array if it's not already
    x_array = np.atleast_1d(x)
    scalar_input = np.isscalar(x)
    
    # Handle the case of n = 0 directly
    if n == 0:
        result = np.array([f(xi, *params) for xi in x_array])
        return result[0] if scalar_input else result
    
    # Apply the formula for each x value
    result = np.zeros_like(x_array, dtype=float)
    
    for i, xi in enumerate(x_array):
        # Compute the sum part of the formula
        sum_part = 0.0
        for k in range(n + 1):
            # Calculate (-1)^(k+n) * binomial(n,k) * f(x+kh)
            coefficient = (-1)**(k+n) * comb(n, k, exact=True)
            sum_part += coefficient * f(xi + k*h, *params)
        
        # Divide by h^n to get the derivative
        result[i] = sum_part / (h**n)
    
    # Return scalar if input was scalar
    return result[0] if scalar_input else result


# Example usage
if __name__ == "__main__":
    # Example 1: Calculate derivatives of sin(x)
    def sin_func(x):
        return np.sin(x)
    
    x_points = np.linspace(0, 2*np.pi, 5)
    
    # First derivative of sin(x) is cos(x)
    print("First derivative of sin(x):")
    print("Numerical:", Dn(sin_func, 1, x_points))
    print("Analytical:", np.cos(x_points))
    print()
    
    # Second derivative of sin(x) is -sin(x)
    print("Second derivative of sin(x):")
    print("Numerical:", Dn(sin_func, 2, x_points))
    print("Analytical:", -np.sin(x_points))
    print()
    
    # Third derivative of sin(x) is -cos(x)
    print("Third derivative of sin(x):")
    print("Numerical:", Dn(sin_func, 3, x_points))
    print("Analytical:", -np.cos(x_points))
    print()
    
    # Example 2: Function with parameters
    def polynomial(x, a, b, c):
        return a*x**2 + b*x + c
    
    # First derivative of ax^2 + bx + c is 2ax + b
    a, b, c = 3, 2, 1
    x_val = 2.0
    
    print(f"First derivative of {a}x^2 + {b}x + {c} at x = {x_val}:")
    print("Numerical:", Dn(polynomial, 1, x_val, a, b, c))
    print("Analytical:", 2*a*x_val + b)
    print()
    
    # Second derivative of ax^2 + bx + c is 2a
    print(f"Second derivative of {a}x^2 + {b}x + {c} at x = {x_val}:")
    print("Numerical:", Dn(polynomial, 2, x_val, a, b, c))
    print("Analytical:", 2*a)
    print()
    
    # Show effects of step size h
    print("Effect of step size h on accuracy (second derivative of polynomial):")
    for h_val in [1e-1, 1e-3, 1e-5, 1e-7]:
        result = Dn(polynomial, 2, x_val, a, b, c, h=h_val)
        error = abs(result - 2*a)
        print(f"h = {h_val:.1e}, result = {result}, error = {error:.2e}")