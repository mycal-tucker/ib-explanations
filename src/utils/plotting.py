import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np
from ib_color_naming.src.figures import WCS_CHIPS
from ib_color_naming.src.tools import lab2rgb
from sklearn.decomposition import PCA

import src.settings as settings
from ib_color_naming.src.figures import get_color_grid, grid2img


def plot_pca(comms, coloring_data, legend=None, sizes=None, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    if settings.pca is None:
        settings.pca = PCA(n_components=2)
        settings.pca.fit(comms)
    if comms.shape[1] > 2:
        transformed = settings.pca.transform(comms)
    else:
        transformed = comms
    s = 20 if sizes is None else sizes
    pcm = ax.scatter(transformed[:, 0], transformed[:, 1], s=s, marker='o', c=coloring_data)
    if legend is not None:
        handles, labels = pcm.legend_elements(prop='colors')
        ax.legend(handles, legend)
    ax.title.set_text("a. Communication Vectors (2D PCA)")


def plot_2d_scatter(x_vals, y_vals, filename=None):
    idxs = np.arange(len(x_vals))
    scatter = plt.scatter(x_vals, y_vals, c=idxs)
    plt.xlabel("Complexity (bits)")
    plt.ylabel("Num abstractions")
    cbar = plt.colorbar(scatter, format=tick.FormatStrFormatter('%.2f'))
    # cbar.set_label('KL Loss Weight', rotation=270, labelpad=15)
    plt.savefig(filename)
    plt.close()


def plot_modemap(prob_array, ib_model, title, filename=None):
    avg_colors = ib_model.mode_map(prob_array)
    # plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300)
    return avg_colors


def plot_comms_pca(pw_m, speaker, save_path=None, ax=None):
    n = pw_m.shape[0]
    pM = np.ones((n, 1)) / n
    qMW = pw_m * pM
    pW = qMW.sum(axis=0)[:, None]
    pC_W = qMW.T / (pW + 1e-20)
    avg_color_per_comm_id = lab2rgb(pC_W.dot(WCS_CHIPS))
    # Now just look up the actual comms associated with each word
    comm_list = []
    coloring_list = []
    sizes = []
    num_abs = 0
    for comm_id in range(speaker.num_tokens):
        comm_prob = pW[comm_id]
        if comm_prob < 0.01:
            continue  # Skip comm id that almost never shows up.
        num_abs += 1
        avg_color = avg_color_per_comm_id[comm_id]
        vec = speaker.vq_layer.prototypes[comm_id].detach().cpu().numpy()
        for _ in range(int(comm_prob * 100)):  # Repeat the vectors proportional to likelihood.
            comm_list.append(vec)
            coloring_list.append(avg_color)
            sizes.append(2000 * comm_prob)  # And rescale plotted points proportional to likelihood.
    if len(comm_list) > 0:
        comm_list = np.vstack(comm_list)
        coloring_list = np.vstack(coloring_list)
        plot_pca(comm_list, coloring_list, sizes=sizes, ax=ax)
    plt.savefig(save_path + '/pca.png')
    return num_abs

