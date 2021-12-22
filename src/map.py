import numpy as np
import matplotlib.pyplot as plt


class path_map():
    def __init__(self, x_limits, y_limits, resolution):
        self.mapping = np.array(
            [[0 for j in range(resolution[0])] for i in range(resolution[1])])
        self.x_lim = x_limits
        self.y_lim = y_limits
        self.res = resolution
        self.res_x = (self.x_lim[1] - self.x_lim[0]) / resolution[0]
        self.res_y = (self.y_lim[1] - self.y_lim[0]) / resolution[1]

    def new_index(self, z):
        m = int((z.real - self.x_lim[0]) /
                (self.x_lim[1] - self.x_lim[0]) * self.res[0])
        n = int((z.imag - self.y_lim[0]) /
                (self.y_lim[1] - self.y_lim[0]) * self.res[1])
        return (m, n)

    def draw_line(self, z1, z2, num=1):
        x1 = (z1.real - self.x_lim[0]) / (self.x_lim[1] - self.x_lim[0])
        y1 = (z1.imag - self.y_lim[0]) / (self.y_lim[1] - self.y_lim[0])
        x2 = (z2.real - self.x_lim[0]) / (self.x_lim[1] - self.x_lim[0])
        y2 = (z2.imag - self.y_lim[0]) / (self.y_lim[1] - self.y_lim[0])
        s = int((x2 - x1) * self.res[0]) + int((y2 - y1) / self.res[1]) + 1
        for i in range(2 * s):
            m = int((x1 * i / (2 * s) + x2 * (1 - i / (2 * s))) * self.res[0])
            n = int((y1 * i / (2 * s) + y2 * (1 - i / (2 * s))) * self.res[1])
            if m >= 0 and m < self.res[0] and n >= 0 and n < self.res[1]:
                self.mapping[m, n] = num
        if s == 1:
            return (m, n)
        else:
            return (-1, -1)

    def check_intersection(self, z1, z2, num=1):
        intersections = []
        x1 = (z1.real - self.x_lim[0]) / (self.x_lim[1] - self.x_lim[0])
        y1 = (z1.imag - self.y_lim[0]) / (self.y_lim[1] - self.y_lim[0])
        x2 = (z2.real - self.x_lim[0]) / (self.x_lim[1] - self.x_lim[0])
        y2 = (z2.imag - self.y_lim[0]) / (self.y_lim[1] - self.y_lim[0])
        s = int((x2 - x1) * self.res[0]) + int((y2 - y1) / self.res[1]) + 1
        for i in range(2 * s):
            m = int((x1 * i / (2 * s) + x2 * (1 - i / (2 * s))) * self.res[0])
            n = int((y1 * i / (2 * s) + y2 * (1 - i / (2 * s))) * self.res[1])
            if m >= 0 and m < self.res[0] and n >= 0 and n < self.res[1]:
                if self.mapping[m, n] != 0:
                    if (m, n) != self.new_index(z1):
                        intersections.append((m, n))
        return intersections

    def get_coordinates(self, ind):
        return self.x_lim[0] + ind[0] * self.res_x + 1j * (
            self.y_lim[0] + ind[1] * self.res_y)

    def show_map(self, filename="map"):
        fig = plt.figure(figsize=(15, 10), dpi=200)
        plt.imshow(1 - self.mapping.T, cmap='Greys_r')
        fig.savefig("../graphics/" + filename + ".png", dpi=200)
