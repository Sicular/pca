import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from remote_plot import plt as pltr

def gram_schmidt(w):
    q, r = np.linalg.qr(w)
    return q

main_fig = plt.figure(figsize=(10,10))
def display3D(input, mu, dependency, animation = False):
    if animation:
        main_fig.clear()
        fig = main_fig
    else:
        fig = plt.figure()

    # Plot the ellipse with zorder=0 in order to demonstrate
    # its transparency (caused by the use of alpha).
    ax_kwargs = fig.add_subplot(projection="3d")
    # ax_kwargs.axvline(c="grey", lw=1)
    # ax_kwargs.axhline(c="grey", lw=1)
    ax_kwargs.scatter(input[:, 0], input[:, 1], input[:, 2], s=0.5)
    ax_kwargs.scatter(mu[0], mu[1], mu[2], c="red", s=3)
    ax_kwargs.set_xlim((-10,10))
    ax_kwargs.set_ylim((-10,10))
    ax_kwargs.set_zlim((-10,10))
    for i in range(dependency.shape[0]):
        size = 15 / np.linalg.norm(dependency[i])
        ax_kwargs.plot(
            [-dependency[i][0] * size, dependency[i][0] * size],
            [-dependency[i][1] * size, dependency[i][1] * size],
            [-dependency[i][2] * size, dependency[i][2] * size],
            "g-",
        )
    ax_kwargs.set_title("Using keyword arguments")

    # fig.subplots_adjust(hspace=0.25)
    if not animation:
        plt.show()
    else:
        plt.pause(0.01)
# pltr.stop_server()
# pltr.start_server()
pltr.figure()
main_fig = pltr._figure
def display3D_remote(input, mu, dependency, animation = False):
    if animation:
        main_fig.clear()
        fig = main_fig
    else:
        fig = plt.figure()

    # Plot the ellipse with zorder=0 in order to demonstrate
    # its transparency (caused by the use of alpha).
    ax_kwargs = fig.add_subplot(projection="3d")
    # ax_kwargs.axvline(c="grey", lw=1)
    # ax_kwargs.axhline(c="grey", lw=1)
    ax_kwargs.scatter(input[:, 0], input[:, 1], input[:, 2], s=0.5)
    ax_kwargs.scatter(mu[0], mu[1], mu[2], c="red", s=3)
    ax_kwargs.set_xlim((-10,10))
    ax_kwargs.set_ylim((-10,10))
    ax_kwargs.set_zlim((-10,10))
    for i in range(dependency.shape[0]):
        size = 15 / np.linalg.norm(dependency[i])
        ax_kwargs.plot(
            [-dependency[i][0] * size, dependency[i][0] * size],
            [-dependency[i][1] * size, dependency[i][1] * size],
            [-dependency[i][2] * size, dependency[i][2] * size],
            "g-",
        )
    ax_kwargs.set_title("Using keyword arguments")

    # fig.subplots_adjust(hspace=0.25)
    pltr.show()


def display2D(input, mu, dependency):
    fig = plt.figure()

    # Plot the ellipse with zorder=0 in order to demonstrate
    # its transparency (caused by the use of alpha).
    ax_kwargs = fig.add_subplot()
    # ax_kwargs.axvline(c="grey", lw=1)
    # ax_kwargs.axhline(c="grey", lw=1)
    ax_kwargs.scatter(input[:, 0], input[:, 1], s=0.5)
    ax_kwargs.scatter(mu[0], mu[1], c="red", s=3)
    for i in range(dependency.shape[0]):
        size = 15 / np.linalg.norm(dependency[i])
        ax_kwargs.plot(
            [-dependency[i][0] * size, dependency[i][0] * size],
            [-dependency[i][1] * size, dependency[i][1] * size],
            "g-",
        )
    ax_kwargs.set_title("Using keyword arguments")

    fig.subplots_adjust(hspace=0.25)
    plt.show()


def get_correlated_dataset(n, dependency, mu, scale, dim=3):
    # np.random.seed(42)
    latent = np.random.randn(n, dim)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    print(scaled_with_offset)
    return scaled_with_offset


def get_random_dataset(n, m):
    # np.random.seed(42)
    return np.random.randn(n, m)

