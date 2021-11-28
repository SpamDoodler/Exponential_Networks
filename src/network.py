import sw_curve as sw
import diffeq as dq
import moebius as mb

# Python libraries
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import map as mp


class path():
    def __init__(self, x, y_i, y_j):
        self.x = np.array(x)
        self.y_i = np.array(y_i)
        self.y_j = np.array(y_j)


class network():
    def __init__(self, H,  dt, steps, theta, expo=False, comp=False):
        # Indices of the network are
        # [time, roots, trajectory starting from branch points]
        self.maps = np.array(
            [mp.path_map((-2, 2), (-2, 2), (2000, 2000))
             for i in range(3)])
        self.curve = sw.sw_curve(H, comp=comp)
        self.expo = expo
        self.dt = dt
        self.steps = steps
        self.theta = theta
        # self.path_x = np.array([])
        # self.path_y_i = np.array([])
        # self.path_y_j = np.array([])
        self.x = np.array([[[0j for k in range(3)]
                            for j in range(len(self.curve.branch_points))]
                           for i in range(self.steps)])

        self.y_i = np.array([[[0j for k in range(3)]
                              for j in range(len(self.curve.branch_points))]
                             for i in range(self.steps)])

        self.y_j = np.array([[[0j for k in range(3)]
                              for j in range(len(self.curve.branch_points))]
                             for i in range(self.steps)])

    def start_paths(self, n=0):
        for j in range(len(self.curve.branch_points)):
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
                        # self.path_y_i = np.append(
                        # self.path_y_i, np.array(complex(rts[i])))
                        # self.path_y_j = np.append(
                        # self.path_y_i, np.array(complex(rts[i])))
                        self.y_i[0, j, k] = complex(rts[i])
                        self.y_j[0, j, k] = complex(rts[i])
                    break
            for k in range(3):
                self.x[0, j, k] = self.curve.branch_points[j]

        for j in range(len(self.curve.branch_points)):
            kappa = -1 / 2 * self.curve.d2Hy2(
                self.curve.branch_points[j], self.y_i[0, j, k]) / \
                self.curve.dHx(self.x[0, j, k], self.y_i[0, j, k])
            if self.expo:
                dx = ((3 / 4 * np.sqrt(kappa) * self.y_i[0, j, k] *
                       self.x[0, j, k] *
                       np.exp(1j * self.theta) * self.dt)**2)**(1 / 3)
            else:
                dx = ((3 / 4 * np.sqrt(kappa) * self.x[0, j, k]
                       * np.exp(1j * self.theta) * self.dt)**2)**(1 / 3)

            for k in range(3):
                dx_k = dx*np.exp(2 * np.pi * k * 1j / 3)
                self.x[1, j, k] = self.x[0, j, k] + dx_k
                self.maps[k].draw_line(self.x[0, j, k], self.x[1, j, k])
                if self.expo:
                    self.y_i[0, j, k] = np.log(self.y_i[0, j, k])
                    self.y_j[0, j, k] = self.y_i[0, j, k]
                    self.y_i[1, j, k] = self.y_i[0, j, k] \
                        + ((3 / 4 * np.sqrt(kappa) * self.x[0, j, k]
                            * np.exp(1j * self.theta) * self.dt))**(1 / 3)\
                        * np.exp(2 * np.pi * k * 1j / 3) / np.sqrt(kappa)
                    self.y_j[1, j, k] = self.y_i[0, j, k] \
                        - ((3 / 4 * np.sqrt(kappa) * self.x[0, j, k]
                            * np.exp(1j * self.theta) * self.dt))**(1 / 3)\
                        * np.exp(2 * np.pi * k * 1j / 3) / np.sqrt(kappa)

                else:
                    self.y_i[1, j, k] = self.y_i[1, j, k] \
                        + np.sqrt(dx_k) / np.sqrt(kappa)
                    self.y_j[1, j, k] = self.y_i[1, j, k] \
                        - np.sqrt(dx_k) / np.sqrt(kappa)
                if self.expo:
                    pass  # print(self.y_j[1, j, k] - self.y_i[1, j, k])

    def plot_network(self,
                     steps=0, paths=[], points=[], fix_axis=True,
                     filename="Network.png"):
        if steps == 0:
            steps = self.steps
        fig = plt.figure(figsize=(15, 10), dpi=200)
        if fix_axis:
            plt.xlim(-2, 2)
            plt.ylim(-1.7, 1.7)
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
        fig.savefig("../graphics/" + filename + ".png", dpi=200)
        if self.expo:
            A = np.array([[1, 0], [-1, 1/4]])
            fig2 = plt.figure(figsize=(15, 10), dpi=200)
            for j in points:
                for k in paths:
                    x_tf = mb.moebius(A, self.x[:steps, j, k])
                    plt.plot(x_tf.real, x_tf.imag, '-', color='black')
            marks = np.array([-1 + 0j, 0 + 0j])
            bp_tf = mb.moebius(A, self.curve.branch_points)
            plt.plot(bp_tf.real, bp_tf.imag, 'x', color='orange')
            plt.plot(marks.real, marks.imag, 'bo')
            fig2.savefig("../graphics/" + filename + "_transformed.png",
                         dpi=200)

    def evolve(self, steps=0):
        dt = self.dt
        if steps == 0:
            steps = self.steps

        if steps >= self.steps:
            steps = self.steps

        signs = np.array([[0 for k in range(3)]
                          for j in range(len(self.curve.branch_points))])
        last_index = [(0, 0), (0, 0), (0, 0)]
        new_index = last_index
        for k in range(3):
            new_index[k] = self.maps[k].new_index(self.x[2, 0, k])
        for i in range(2, steps):
            for j in range(len(self.curve.branch_points)):
                for k in range(3):
                    x = self.x[i - 1, j, k]
                    y1 = self.y_i[i - 1, j, k]
                    y2 = self.y_j[i - 1, j, k]
                    # dy = dq.rk4([x, y1, y2], self.dt, self.theta)
                    dy = dq.rk4(self.curve.sw_diiferential,
                                [x, y1, y2], dt,
                                self.theta, expo=self.expo)
                    # self.curve.sw_diiferential([x, y1, y2], self.theta)*dt
                    if i == 2:
                        if (dy[0].real * (x - self.x[0, j, k]).real +
                                dy[0].imag*(x - self.x[0, j, k]).imag) > 0:
                            signs[j, k] = 1
                        else:
                            signs[j, k] = 1
                    signs[0, 0] = 1
                    signs[0, 1] = 1
                    signs[0, 2] = 1
                    self.x[i, j, k] = x + signs[j, k]*dy[0]
                    self.y_i[i, j, k] = y1 + signs[j, k] * dy[1]
                    self.y_j[i, j, k] = y2 + signs[j, k] * dy[2]
                    intersections = []
                    last_index[k] = new_index[k]
                    new_index[k] = self.maps[k].new_index(self.x[i, j, k])
                    if new_index[k] != last_index[k]:
                        for g in range(3):
                            intersections.append(self.maps[g].check_intersection(
                                self.x[i - 1, j, k], self.x[i, j, k]))
                        self.maps[k].draw_line(self.x[i - 1, j, k], self.x[i - 1, j, k])
                        if len(intersections) != 0:
                            print(intersections)
                    # print(self.curve.H(5j, 5))
                    # if i % 1000 == 0:
                    # if i % 1000 == 0:
                    # print("Old", i,
                    # self.curve.H(
                    # self.x[i, j, k],
                    # np.exp(self.y_i[i, j, k])))
                    # if False and abs(self.x[i, j, k]) > 0.00001:
                    # rts_dict = sym.roots(
                    # self.curve.H_sym.subs(
                    # self.curve.x_sym, self.x[i, j, k]))
                    # rts = []
                    # for rt in rts_dict:
                    # for whatever in range(rts_dict[rt]):
                    # rts.append(complex(rt))
                    # a = np.argmin(
                    # [abs(rts[t] - np.exp(self.y_i[i, j, k]))
                    # for t in range(len(rts))])
                    # b = np.argmin([abs(rts[t] - np.exp(self.y_j[i, j, k]))
                    # for t in range(len(rts))])
                    # #n_a = int(
                    # (np.log(rts[a]) - self.y_i[i, j, k] + 1j * np.pi).imag
                    # #          / (2 * np.pi))
                    # #n_b = int(
                    # (np.log(rts[b]) - self.y_j[i, j, k] + 1j * np.pi).imag
                    # #          / (2 * np.pi))
                    # # print("y_i", np.log(rts[a]) - self.y_i[i, j, k], n_a)
                    # # print("y_j", np.log(rts[b]) - self.y_j[i, j, k], n_b)
                    # # self.y_i[i, j, k] = np.log(rts[a]) - 2j * np.pi * n_a
                    # # self.y_j[i, j, k] = np.log(rts[b]) - 2j * np.pi * n_b
                    if i % 1000 == 0:
                        print(i,
                              self.curve.H(
                                  self.x[i, j, k],
                                  np.exp(self.y_i[i, j, k])))
                if i % 1000 == 0:
                    if dt <= 0.0001:
                        dt = dt*10
