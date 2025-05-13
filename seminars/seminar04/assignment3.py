# %% Problem 1
#
#A particle of mass \( m = 2 kg \) is moving with a velocity \( \vec{v} = (4 m/s)\hat{i}−(3 m/s)\hat{j}​ \). At a particular instant, its position vector relative to the origin is \( \vec{r}= (6 m)\hat{i}+(3 m)\hat{j}−(2 m)\hat{k}\). Note that \( \hat{i}, \hat{j}​,\hat{k}\) indicate the unit vectors, e.g. of a cartesian coordinate system.​​

# Calculate the particle's linear momentum vector \(\vec{p}\).
# Using the cross product, compute the angular momentum \(\vec{L}\).
# Do not print anything. Import and use numpy. Use the following variables r, p, m, v, L.
#
#
import numpy as np

m = 2
v = np.array([4, -3, 0])  # velocity in m/s
r = np.array([6, 3, -2])  # position vector in m

p = m * v

L = np.cross(r, p)
L



# %% Problem 2
#
# Given the following matrices:
# \(
# A = \begin{pmatrix}
# 9 & -2 & 8 \\
# -2 & 6 & -4 \\
# 8 & -4 & 26
# \end{pmatrix}
# \), \(
# B = \begin{pmatrix}
# 10 & 0 & -2i \\
# 0 & 10 & i \\
# 2i & -i & 14
# \end{pmatrix}
# \)

# Write a Python program to: 

# Compute the eigenvalues and eigenvectors of \( A \) and \( B \) and store them correspondingly in variables eigenvalues_A, eigenvectors_A and eigenvalues_B, eigenvectors_B.
# Compute the commutator \( [A,B] \) and store it in variable C.
import numpy as np

A = np.array([[9, -2, 8],
              [-2, 6, -4],
              [8, -4, 26]])

B = np.array([[10, 0, -2j],
              [0, 10, 1j],
              [2j, -1j, 14]])

eigenvalues_A, eigenvectors_A = np.linalg.eig(A)
eigenvalues_B, eigenvectors_B = np.linalg.eig(B)

# [A, B] = AB - BA
AB = A @ B
BA = B @ A
C = AB - BA

C

# %% Problem 3
#
# When light travels from one medium to another, it bends according to Snell's Law:
#
# \( n_1 \sin(\theta_1) = n_2 \sin(\theta_2) \)
#
# where:
# - \( n_1 \)is the refractive index of the first medium
# - \( n_2 \) is the refractive index of the second medium
# - \( \theta_1 \) is the angle of incidence (in radians)
# - \( \theta_2 \) is the angle of refraction (in radians)

# Write a python function snell that calculates the angle of refraction \( \theta_2 \). The function should contain the parameters \( n_1 \), \( n_2 \) and \( \theta_1 \). The input angle should be supplied in degrees. The function should return the value of \( \theta_2 \) in degrees. Import numpy as np.

import numpy as np

def snell(n1, n2, theta1):
    theta2=np.arcsin(n1*np.sin(np.deg2rad(theta1))/n2)
    return np.rad2deg(theta2)
snell(1.5,1.,41.81)
