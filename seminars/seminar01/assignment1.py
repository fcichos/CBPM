# %% import modules
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12,
                     'lines.linewidth': 1,
                     'lines.markersize': 10,
                     'axes.labelsize': 11,
                     'xtick.labelsize' : 10,
                     'ytick.labelsize' : 10,
                     'xtick.top' : True,
                     'xtick.direction' : 'in',
                     'ytick.right' : True,
                     'ytick.direction' : 'in',
                     'figure.figsize': (4, 3),
                     'figure.dpi': 150})

def get_size(w,h):
      return((w/2.54,h/2.54))

# %% define function
def lennard_jones(r, epsilon=1, sigma=1):
    return 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)


# %% do plotting

sig=1
eps=1

r = np.linspace(0.8, 3, 1000)
V = lennard_jones(r,epsilon=eps,sigma=sig)

plt.figure(figsize=get_size(8, 6),dpi=150)
plt.plot(r/sig, V/eps, 'b-', linewidth=2)
plt.grid(True)
plt.xlabel('r/σ')
plt.ylabel('V/ε')
plt.title('Lennard-Jones Potential')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.ylim(-1.5, 3)
plt.show()
# %% Velocity Verlet
#
# Parameters

g = 9.81  # m/s^2
dt = 0.5  # time step
t_max = 2.0  # total simulation time
steps = int(t_max/dt)

# Initial conditions
y0 = 20.0  # initial height
v0 = 0.0   # initial velocity


# Arrays to store results
t = np.zeros(steps)
y = np.zeros(steps)
v = np.zeros(steps)
a = np.zeros(steps)

# Initial values
y[0] = y0
v[0] = v0
a[0] = -g

# Velocity Verlet integration
for i in range(1, steps):
    t[i] = i * dt
    y[i] = y[i-1] + v[i-1] * dt + 0.5 * a[i-1] * dt**2  # update position
    a_new = -g                                          # new acceleration (assuming constant gravity)
    v[i] = v[i-1] + 0.5 * (a[i-1] + a_new) * dt         # update velocity
    a[i] = a_new                                        # store new acceleration

y_analytical = y0 + v0*t - 0.5*g*t**2
plt.figure(figsize=get_size(8, 6), dpi=150)
plt.plot(t, y)
plt.plot(t, y_analytical, 'r--')

plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.title('Free Fall Motion')
plt.grid(True)
plt.show()
