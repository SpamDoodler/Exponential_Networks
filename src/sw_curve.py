import sympy as sym
import numpy as np


# Define the defining equation for the Seiberg-Witten curve
# Example 1: GMN-
def phi2(z):
    return 2 / (z - 1)**2 + 2 / (z + 1)**2 - 4 / z**2


def phi3(z):
    return ((1 - 3*(z - 1) + 6*(z - 1)**2) / (z - 1)**3 +
            (1 + 3*(z + 1) +
            6*(z + 1)**2) / (z + 1)**3 - (2 + 12*z**2) / z**3
            )


def phi(z):
    return np.array([1, 0, phi2(z), phi3(z)])


# Example 2: Exponential network
def H_c3(x, y):
    return -x + y**2 + y


# The Seiberg Witten curve class x: Basespace coordiate, y: fiber coordinate
class sw_curve():
    def __init__(self, H, comp=True):
        self.x_sym, self.y_sym = sym.symbols('x y')
        if comp:
            self.n = len(H(self.x_sym))
            self.H_sym = sym.simplify(np.sum([H(self.x_sym)[i] *
                                              self.y_sym**(self.n - 1 - i)
                                              for i in range(self.n)]))
        else:
            self.H_sym = sym.simplify(H(self.x_sym, self.y_sym))
        self.n_y, self.d_y = sym.fraction(sym.together(self.H_sym), self.y_sym)
        self.H = sym.lambdify([self.x_sym, self.y_sym],
                              self.H_sym, "numpy")

        self.dHx_sym = sym.diff(self.H_sym, self.x_sym)
        self.dHy_sym = sym.diff(self.H_sym, self.y_sym)
        self.d2Hy2_sym = sym.diff(self.dHy_sym, self.y_sym)

        self.dHx = sym.lambdify([self.x_sym, self.y_sym],
                                self.dHx_sym, "numpy")
        self.dHy = sym.lambdify([self.x_sym, self.y_sym],
                                self.dHy_sym, "numpy")
        self.d2Hy2 = sym.lambdify([self.x_sym, self.y_sym],
                                  self.d2Hy2_sym, "numpy")
        self.branch_points, self.sing_points = self.branch_singular_points()

    def branch_singular_points(self):
        disc_poly_sym = sym.Poly(
            self.H_sym*self.d_y, self.y_sym).discriminant()
        n, d = sym.fraction(sym.together(disc_poly_sym))
        sing_sym = sym.solve(sym.Poly(d, self.x_sym))
        sing = np.array([sing_sym[i].evalf() for i in range(len(sing_sym))])
        branch_coeffs = sym.Poly(
            sym.simplify(d*disc_poly_sym), self.x_sym).all_coeffs()
        branch = np.roots(branch_coeffs)
        return branch, sing

    def sw_diiferential(self, pt, theta, expo=False):
        x = pt[0]
        y1 = pt[1]
        y2 = pt[2]
        if not(expo):
            dx = x * np.exp(1j * theta) / (y2 - y1)
            dy1 = -self.dHx(x, y1) / self.dHy(x, y1) * dx
            dy2 = -self.dHx(x, y2) / self.dHy(x, y2) * dx
        else:
            # print(y1, y2, x, theta)
            dx = x * np.exp(1j * theta) / \
                (y2 - y1)
            # print(dx)
            dy1 = -self.dHx(x, np.exp(y1)) / \
                self.dHy(x, np.exp(y1)) * dx / np.exp(y1)
            dy2 = -self.dHx(x, np.exp(y2)) / \
                self.dHy(x, np.exp(y2)) * dx / np.exp(y2)
        return np.array([dx, dy1, dy2])
