import sw_curve as sw
import network as sn
import numpy as np
# import matplotlib.pyplot as plt


def main():
    # network = sn.network(lambda z: sw.phi(z), 1e-6, 5000, np.pi / 4, comp=True)
    # network.start_paths()
    # network.evolve()
    # network.plot_network()
    expo_net = sn.network(sw.H_c3, 1e-8, 13000, 0*np.pi, expo=True)
    expo_net.start_paths()
    expo_net.evolve()
    expo_net.plot_network(paths=[0,2], filename="expo_net.png")


if __name__ == '__main__':
    main()
