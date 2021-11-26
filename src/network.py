import sw_curve as sw
import diffeq as dq

# Python libraries
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt


class network():
    def __init__(self, H,  dt, steps, theta, expo=False, comp=False):
        # Indices of the network are
        # [time, roots, trajectory starting from branch points]
        self.curve = sw.sw_curve(H, comp=comp)
        self.expo = expo
        self.dt = dt
        self.steps = steps
        self.theta = theta
        self.x = np.array([[[0j for k in range(3)]
                            for j in range(len(self.curve.branch_points))]
                           for i in range(self.steps)])

        self.y_i = np.array([[[0j for k in range(3)]
                              for j in range(len(self.curve.branch_points))]
                             for i in range(self.steps)])

        self.y_j = np.array([[[0j for k in range(3)]
                              for j in range(len(self.curve.branch_points))]
                             for i in range(self.steps)])

    def start_paths(self):
        for j in range(len(self.curve.branch_points)):
            print(self.curve.branch_points)
            rts_dict = sym.roots(
                (self.curve.d_y*self.curve.H_sym).subs(
                    self.curve.x_sym, self.curve.branch_points[j]))
            rts = []
            for rt in rts_dict:
                for i in range(rts_dict[rt]):
                    rts.append(rt)

            for i in range(len(rts) - 1):
                if abs(np.prod
                       ([rts[i] - rts[u] for u in range(i + 1, len(rts))])
                       ) < 0.0001:
                    for k in range(3):
                        self.y_i[0, j, k] = rts[i]
                        self.y_j[0, j, k] = rts[i]
                    break
            print(rts)
            for k in range(3):
                self.x[0, j, k] = self.curve.branch_points[j]

        for j in range(len(self.curve.branch_points)):
            kappa = -1 / 2 * self.curve.d2Hy2(
                self.curve.branch_points[j], self.y_i[0, j, k]) / \
                self.curve.dHx(self.x[0, j, k], self.y_i[0, j, k])
            print(kappa)
            dx = ((3 / 4 * np.sqrt(kappa)
                   * self.x[0, j, k] * np.exp(1j * self.theta
                                              ) * self.dt)**2)**(1 / 3)

            for k in range(3):
                dx_k = dx*np.exp(2 * np.pi * k * 1j / 3)
                self.x[1, j, k] = self.x[0, j, k] + dx_k
                self.y_i[1, j, k] = self.y_i[0, j, k] \
                    + np.sqrt(dx_k) / np.sqrt(kappa)
                self.y_j[1, j, k] = self.y_j[0, j, k] \
                    - np.sqrt(dx_k) / np.sqrt(kappa)

    def plot_network(self,
                     steps=0, paths=[], points=[], filename="Network.png"):
        if steps == 0:
            steps = self.steps
        fig = plt.figure(figsize=(15, 10), dpi=200)
        plt.xlim(-1.6, 1.5)
        plt.ylim(-1.3, 1.3)
        if len(points) == 0:
            points = np.arange(len(self.curve.branch_points))

        if len(paths) == 0:
            paths = np.arange(3)

        for j in points:
            for k in paths:
                plt.plot(self.x[:steps, j, k].real, self.x[:steps, j, k].imag,
                         '-', color='black')
        plt.plot(self.curve.branch_points.real, self.curve.branch_points.imag,
                 'x', color='orange')
        plt.plot(self.curve.sing_points.real, self.curve.sing_points.imag,
                 'b.')
        fig.savefig("../graphics/" + filename, dpi=200)

    def evolve(self, steps=0, n=0):
        dt = self.dt
        if steps == 0:
            steps = self.steps

        if steps >= self.steps:
            steps = self.steps

        signs = np.array([[0 for k in range(3)]
                          for j in range(len(self.curve.branch_points))])

        for i in range(2, steps):
            for j in range(len(self.curve.branch_points)):
                for k in range(3):
                    x = self.x[i - 1, j, k]
                    y1 = self.y_i[i - 1, j, k]
                    y2 = self.y_j[i - 1, j, k]
                    # dy = dq.rk4([x, y1, y2], self.dt, self.theta)
                    dy = dq.rk4(self.curve.sw_diiferential,
                                [x, y1, y2], dt,
                                self.theta, expo=self.expo, n=n)
                    # self.curve.sw_diiferential([x, y1, y2], self.theta)*dt
                    if i == 2:
                        if (dy[0].real * (x - self.x[0, j, k]).real +
                                dy[0].imag*(x - self.x[0, j, k]).imag) > 0:
                            signs[j, k] = 1
                        else:
                            signs[j, k] = -1
                    self.x[i, j, k] = x + signs[j, k]*dy[0]
                    self.y_i[i, j, k] = y1 + signs[j, k] * dy[1]
                    self.y_j[i, j, k] = y2 + signs[j, k] * dy[2]
            if i % 1000 == 0:
                if dt <= 0.0001:
                    dt = dt*100
