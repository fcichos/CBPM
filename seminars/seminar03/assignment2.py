# %% problem 1
# Write python code to import the numpy module under the namespace np.
import numpy as np

# %% problem 2
# Write Python code to import the sqrt function from the math module such that it can be used under the function name square_root.
from math import sqrt as square_root

square_root(2)



# %% problem 3
# Define a function year_to_century that returns the century for a given year. The first century spans from the year 1 up to and including the year 100, the second - from the year 101 up to and including the year 200, etc.

def year_to_century(year):
    return (year + 99) // 100


year_to_century(100)



# %% problem 4
# Define a function triangle that takes the 3 sides of a triangle and returns one of the following: 

# "No Triangle": If the sides don't form a valid triangle (degenerate case).
# "Acute Triangle": If all angles are acute.
# "Right Triangle": If one angle is exactly 90°.
# "Obtuse Triangle": If one angle is obtuse.
#


def triangle(a, b, c):
    sides = sorted([a, b, c]) # Sort sides so that a <= b <= c
    a, b, c = sides

    if a + b <= c:
        return "No Triangle"

    if a**2 + b**2 == c**2:
        return "Right Triangle"

    if a**2 + b**2 > c**2:
        return "Acute Triangle"

    return "Obtuse Triangle"

triangle(3, 4, 6)



# %% problem 5
# A laboratory has two different lasers each of them with multiple wavelength that can be stored in a set. Create a function that finds common wavelengths using sets.




def analyze_wavelengths(laser1_wavelengths: set, laser2_wavelengths: set):

    common_wavelengths = laser1_wavelengths & laser2_wavelengths

    return common_wavelengths


result = analyze_wavelengths({532, 630, 405, 450},{450, 532, 635, 980})
print(result)
