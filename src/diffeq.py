# import numpy as np


def fwd_euler(f, y, dt, theta):
    return dt*f(y, theta)


def rk4(f, y, dt, theta, expo):
    k1 = f(y, theta, expo)
    k2 = f(y + dt * k1 / 2, theta, expo)
    k3 = f(y + dt * k2 / 2, theta, expo)
    k4 = f(y + dt * k3, theta, expo)
    return 1 / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4)
