import sw_curve as sw
import network as sn
# import numpy as np
# import map as mp
# import matplotlib.pyplot as plt


def main():
    # network = sn.network(
    #     lambda z: sw.phi(z), 1e-6, 5000, np.pi / 4, comp=True)
    # network.start_paths()
    # network.evolve()
    # network.plot_network()
    steps = 2000000
    theta = 0.0
    expo_net = sn.network(sw.H_c3, 1e-12, steps, theta, expo=True)
    expo_net.start_paths()
    expo_net.evolve()
    expo_net.plot_network(
        paths=[], fix_axis=True, filename="expo_net")
    expo_net.plot_network(
        paths=[0], fix_axis=True, filename="expo_net0")
    expo_net.plot_network(
        paths=[1], fix_axis=True, filename="expo_net1")
    expo_net.plot_network(
        paths=[2], fix_axis=True, filename="expo_net2")
    expo_net.maps[0].show_map("map0")
    expo_net.maps[1].show_map("map1")
    expo_net.maps[2].show_map("map2")


if __name__ == '__main__':
    main()
