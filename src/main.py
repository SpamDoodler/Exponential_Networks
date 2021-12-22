import sw_curve as sw
import network as sn
# import numpy as np
# import map as mp
import matplotlib.pyplot as plt


def main():
    # network = sn.network(
    #     lambda z: sw.phi(z), 1e-6, 5000, np.pi / 4, comp=True)
    # network.start_paths()
    # network.evolve()
    # network.plot_network()
    steps = 60000
    theta = 0.0
    expo_net = sn.network(sw.H_c3, 1e-12, steps, theta, expo=True)
    expo_net.start_paths()
    expo_net.evolve()
    expo_net.plot_network(
        paths=[], fix_axis=True, filename="expo_net_lq")
    expo_net.plot_network(
        paths=[0], fix_axis=True, filename="expo_net0_lq")
    expo_net.plot_network(
        paths=[1], fix_axis=True, filename="expo_net1_lq")
    expo_net.plot_network(
        paths=[2], fix_axis=True, filename="expo_net2_lq")
    expo_net.maps[0].show_map("map0_lq")
    expo_net.maps[1].show_map("map1_lq")
    expo_net.maps[2].show_map("map2_lq")
    print(expo_net.intersections)
    fig = plt.figure(figsize=(15, 10), dpi=200)
    plt.plot(expo_net.intersections[:, 0], expo_net.intersections[:, 1], '.')
    fig.savefig("../graphics/intersections.png", dpi=200)


if __name__ == '__main__':
    main()
