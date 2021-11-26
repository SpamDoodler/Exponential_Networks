import sw_curve as sw
import network as sn
import numpy as np
# import matplotlib.pyplot as plt


def main():
    # network = sn.network(
    #     lambda z: sw.phi(z), 1e-6, 5000, np.pi / 4, comp=True)
    # network.start_paths()
    # network.evolve()
    # network.plot_network()
    steps = 10000
    expo_net = sn.network(sw.H_c3, 1e-10, steps, 0 * np.pi / 4, expo=True)
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


if __name__ == '__main__':
    main()
