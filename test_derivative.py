import numpy as np
from scipy.special import comb
import math

# Define the derivative function (copy of your implementation)
def Dn(f, n, x, *params, h=1e-5):
    x_array = np.atleast_1d(x)
    scalar_input = np.isscalar(x)
    
    if n == 0:
        result = np.array([f(xi, *params) for xi in x_array])
        return result[0] if scalar_input else result
    
    result = np.zeros_like(x_array, dtype=float)
    
    for i, xi in enumerate(x_array):
        sum_part = 0.0
        for k in range(n + 1):
            coefficient = (-1)**(k+n) * comb(n, k, exact=True)
            sum_part += coefficient * f(xi + k*h, *params)
        
        result[i] = sum_part / (h**n)
    
    return result[0] if scalar_input else result

# ============= TEST FUNCTIONS =============

# Define test functions with known derivatives
def test_polynomial():
    print("\n===== TESTING POLYNOMIAL FUNCTIONS =====\n")
    
    # Test case 1: f(x) = x² + 3x + 2
    def f1(x):
        return x**2 + 3*x + 2
    
    # Analytical derivatives of f1
    def f1_d1(x):
        return 2*x + 3  # First derivative
    
    def f1_d2(x):
        return 2        # Second derivative
    
    def f1_d3(x):
        return 0        # Third derivative
    
    test_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    
    for x in test_values:
        # First derivative
        numerical = Dn(f1, 1, x)
        analytical = f1_d1(x)
        error = abs(numerical - analytical)
        print(f"x = {x}, f'(x): Numerical = {numerical:.10f}, Analytical = {analytical}, Error = {error:.10f}")
        assert error < 1e-5, f"First derivative test failed at x = {x}"
        
        # Second derivative
        numerical = Dn(f1, 2, x)
        analytical = f1_d2(x)
        error = abs(numerical - analytical)
        print(f"x = {x}, f''(x): Numerical = {numerical:.10f}, Analytical = {analytical}, Error = {error:.10f}")
        assert error < 1e-4, f"Second derivative test failed at x = {x}"
        
        # Third derivative (should be zero)
        numerical = Dn(f1, 3, x)
        analytical = f1_d3(x)
        error = abs(numerical - analytical)
        print(f"x = {x}, f'''(x): Numerical = {numerical:.10f}, Analytical = {analytical}, Error = {error:.10f}")
        assert error < 1e-3, f"Third derivative test failed at x = {x}"
        print("")
    
    # Test case 2: f(x) = ax³ + bx² + cx + d with parameters
    def f2(x, a, b, c, d):
        return a*x**3 + b*x**2 + c*x + d
    
    a, b, c, d = 2, -3, 4, -1  # Coefficients
    
    # Test with parameters
    x = 1.5
    
    # First derivative: 3ax² + 2bx + c
    numerical = Dn(f2, 1, x, a, b, c, d)
    analytical = 3*a*x**2 + 2*b*x + c
    error = abs(numerical - analytical)
    print(f"Polynomial with parameters: f'({x}) = {numerical:.10f}, Expected = {analytical}, Error = {error:.10f}")
    assert error < 1e-4, "First derivative with parameters test failed"
    
    # Second derivative: 6ax + 2b
    numerical = Dn(f2, 2, x, a, b, c, d)
    analytical = 6*a*x + 2*b
    error = abs(numerical - analytical)
    print(f"Polynomial with parameters: f''({x}) = {numerical:.10f}, Expected = {analytical}, Error = {error:.10f}")
    assert error < 1e-3, "Second derivative with parameters test failed"
    
    # Third derivative: 6a
    numerical = Dn(f2, 3, x, a, b, c, d)
    analytical = 6*a
    error = abs(numerical - analytical)
    print(f"Polynomial with parameters: f'''({x}) = {numerical:.10f}, Expected = {analytical}, Error = {error:.10f}\n")
    assert error < 1e-2, "Third derivative with parameters test failed"


def test_trigonometric():
    print("\n===== TESTING TRIGONOMETRIC FUNCTIONS =====\n")
    
    # Test sine function and its derivatives
    def f(x):
        return np.sin(x)
    
    test_values = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, np.pi]
    
    for x in test_values:
        # First derivative: cos(x)
        numerical = Dn(f, 1, x)
        analytical = np.cos(x)
        error = abs(numerical - analytical)
        print(f"x = {x:.4f}, sin'(x): Numerical = {numerical:.10f}, Analytical = {analytical:.10f}, Error = {error:.10f}")
        assert error < 1e-4, f"First derivative of sin(x) test failed at x = {x}"
        
        # Second derivative: -sin(x)
        numerical = Dn(f, 2, x)
        analytical = -np.sin(x)
        error = abs(numerical - analytical)
        print(f"x = {x:.4f}, sin''(x): Numerical = {numerical:.10f}, Analytical = {analytical:.10f}, Error = {error:.10f}")
        assert error < 1e-3, f"Second derivative of sin(x) test failed at x = {x}"
        
        # Third derivative: -cos(x)
        numerical = Dn(f, 3, x)
        analytical = -np.cos(x)
        error = abs(numerical - analytical)
        print(f"x = {x:.4f}, sin'''(x): Numerical = {numerical:.10f}, Analytical = {analytical:.10f}, Error = {error:.10f}")
        assert error < 1e-2, f"Third derivative of sin(x) test failed at x = {x}"
        print("")


def test_exponential():
    print("\n===== TESTING EXPONENTIAL FUNCTIONS =====\n")
    
    # Test exponential function: f(x) = e^x
    def f(x):
        return np.exp(x)
    
    test_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    
    for x in test_values:
        # All derivatives of e^x are e^x
        expected = np.exp(x)
        
        for n in range(1, 4):  # Test 1st, 2nd, and 3rd derivatives
            numerical = Dn(f, n, x)
            error = abs(numerical - expected)
            print(f"x = {x}, exp^({n})(x): Numerical = {numerical:.10f}, Analytical = {expected:.10f}, Error = {error:.10f}")
            # Error tolerance grows with derivative order
            tolerance = 1e-4 * (10**(n-1))
            assert error < tolerance, f"{n}th derivative of exp(x) test failed at x = {x}"
        print("")


def test_step_size_sensitivity():
    print("\n===== TESTING STEP SIZE SENSITIVITY =====\n")
    
    # Test the effect of step size on accuracy
    def f(x):
        return x**2
    
    x = 1.0
    true_derivative = 2.0  # f'(x) = 2x
    
    print("First derivative of x² at x = 1.0 with different step sizes:")
    for h in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]:
        result = Dn(f, 1, x, h=h)
        error = abs(result - true_derivative)
        print(f"h = {h:.1e}, result = {result:.12f}, error = {error:.2e}")


def test_vector_input():
    print("\n===== TESTING VECTOR INPUT =====\n")
    
    # Test with vector input
    def f(x):
        return x**2
    
    # Create an array of x values
    x_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # First derivative: 2x
    numerical = Dn(f, 1, x_values)
    analytical = 2 * x_values
    error = np.abs(numerical - analytical)
    
    print("Vector input test for f(x) = x²:")
    print(f"x values: {x_values}")
    print(f"Numerical f'(x): {numerical}")
    print(f"Analytical f'(x): {analytical}")
    print(f"Max error: {np.max(error):.10f}")
    
    assert np.all(error < 1e-4), "Vector input test failed"


# Run all tests
if __name__ == "__main__":
    print("\n===================================================")
    print("TESTING NUMERICAL DERIVATIVE FUNCTION")
    print("===================================================\n")
    
    try:
        # Run all test functions
        test_polynomial()
        test_trigonometric()
        test_exponential()
        test_step_size_sensitivity()
        test_vector_input()
        
        # If we get here, all tests passed
        print("\n===================================================")
        print("✓ ALL TESTS PASSED! The Dn function is working correctly.")
        print("===================================================\n")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")