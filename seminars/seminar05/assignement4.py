# %% Problem 1

# Write a Python function called peak_wavelength that calculates the peak wavelength (in nanometers) of blackbody radiation emitted by an object at a given temperature (in Kelvin), using Wien's Displacement Law.
# Wien's constant is b = 2.897e-3 # m*K.
# What is the peak wavelength at \( T=0~K \)? Handle that case using the keyword raise. Just provide the function. Do not call it. 



def peak_wavelength(T):
    b = 2.897e-3  # m·K (Wien's constant)
    if T <= 0:
        raise ValueError('Invalid Temperature value.')
    return b / T * 1e9



# %% Problem 2

# Define a class with the name particle with a constructor that initializes the property D with twice the value supplied as an argument to the constructor. 


class particle:
    def __init__(self,R):
        self.D=2*R



# %% Problem 3
#
# Assign the 100 numbers between -10 and 10, including -10 and 10, to the variable x using the linspace function of numpy. Assign the function values \( x²-20 \) to the variable y. Store the paired values in the file named "data.txt" in the form "x, y" (e.g. -10, 80) for each line of the file.
#


import numpy as np

x=np.linspace(-10,10,100)
y=x**2-20

with open('data.txt', 'w') as file:
    for x, y in zip(x, y):
        file.write(f"{x}, {y}\n")




# %% Problem 4
# Write a function Dn that calculates the nth derivative of and input function f(x,a,b,...). The derivative function should take the function f as an argument together with the argument x and possible function parameters {a,b,...}. (e.g. def Dn(f, n, x, *params, h=1e-5):)

# The formula for the nth order derivative is 

# f^{(n)}(x)=\lim _{h \rightarrow 0} \frac{1}{h^n} \sum_{k=0}^n(-1)^{k+n}\binom{n}{k} f(x+k h)

# For the numerical estimate of the derivative use a finite value of h. Import the scipy.special.comb() function for calculating the binomial coefficient. Handle the situation when x is just a scalar and not an array. 
#
#
import numpy as np
from scipy.special import comb

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
