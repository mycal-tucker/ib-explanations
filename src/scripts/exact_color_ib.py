import os
import pickle as pkl
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.settings as settings
from embo import InformationBottleneck
from ib_color_naming.src import ib_naming_model
from ib_color_naming.src.figures import grid2img
from src.data.wcs_data import WCSDataset
from src.utils.plotting import plot_modemap


def run_for_setup(label, rew_fn):
    rew_matrix = rew_fn()
    ib = InformationBottleneck(pxy=rew_matrix, alpha=0)
    I_x, I_y, H_m, beta = ib.get_bottleneck()
    abs_marg, encs, decs = ib.get_encs_decs()
    # Now just visualize the abstractions
    prev_num_abs = 0
    save_idx = 0
    distortion_data = []
    comps = []
    infos = []
    eval_labels = ['red_disco', 'green_disco', 'blue_disco', 'red_cont', 'green_cont', 'blue_cont']
    for idx, data in enumerate(zip(encs, decs, I_x, I_y)):
        enc, _, comp, info = data
        if prev_num_abs >= enc.shape[0]:
            continue
        prev_num_abs = enc.shape[0]
        print("Num abs", prev_num_abs)
        comps.append(comp)
        infos.append(info)
        # Just average color to start
        w_m = np.transpose(enc)
        plt.figure(figsize=(2.2, 1.4))  # Adjust width and height as needed
        avg_colors = plot_modemap(w_m, ib_model, title=None)
        base_path = 'saved_data/exact_ib_rgb/train_' + label + '/'
        plt.savefig(base_path + 'viz_avg' + str(save_idx) + '.png', dpi=300)
        plt.close()

        # Now, for each reward function that we care about, color abstractions that way and plot.
        avg_rews = []
        dist_for_checkpoint = []
        for eval_dim in range(7):
            cid_to_rew = _get_cid_to_rew(eval_dim)
            if eval_dim == 6:
                cid_to_rew = get_red_demo_matrix()
            qMW = w_m
            pC = enc.sum(axis=0)[:, None]
            pW_C = np.where(pC > 0, qMW / pC, 1 / qMW.shape[1])
            y = pW_C.argmax(axis=1)
            pW = qMW.sum(axis=0)[:, None]
            pC_W = qMW.T / (pW + 1e-20)
            mu_w = pC_W.dot(cid_to_rew)  # Reward associated with each abstraction
            grid = mu_w[y]
            task_distortion = np.mean((cid_to_rew - grid) ** 2)
            if eval_dim < len(eval_labels):
                dist_for_checkpoint.append(task_distortion)
            grid[pC[:, 0] == 0] = np.nan * grid[pC[:, 0] == 0]
            if len(avg_rews) < 3:
                avg_rews.append(grid)
            # For each color, have the abstraction-based reward. -> [330, 1]
            img = np.flipud(grid2img(grid, small_grid=False))
            # Overwrite the border parts to be -1
            img[:, 1] = -2
            img[0, 1:] = -2
            img[-1, 1:] = -2
            img = np.ma.masked_where(img == -2, img)
            cols = [0, 1] + [2 + i * 3 for i in range(14)]
            subimg = img[:, cols, :]
            plt.figure(figsize=(2.7, 1.4))  # Adjust width and height as needed
            cmap = plt.get_cmap('hot')
            cmap.set_bad(color='white')
            # Recreate the colors based on order, not continuous value.
            unique_values = np.round(np.unique(subimg)[:-1], 3)
            val_to_idx = {}
            for i, val in enumerate(unique_values):
                val_to_idx[val] = i
            ordered_img = subimg.copy()
            for i, row in enumerate(ordered_img):
                for j, val in enumerate(row):
                    val_item = val[0]
                    if isinstance(val_item, float):
                        val_item = np.round(val_item, 3)
                        ordered_img[i, j, 0] = val_to_idx[val_item]
                    else:
                        continue
            # im = plt.imshow(subimg, cmap=cmap, vmin=min_val, vmax=max_val)
            min_val = -0.05
            max_val = len(val_to_idx.keys()) + 0.05
            im = plt.imshow(ordered_img, cmap=cmap, vmin=min_val, vmax=max_val)
            # Overlay white gridlines at every pixel
            for i in range(subimg.shape[0]):
                plt.axhline(y=i - 0.5, color='white', linewidth=0.5)
            for j in range(subimg.shape[1]):
                plt.axvline(x=j - 0.5, color='white', linewidth=0.5)

            plt.ylim([-0.5, subimg.shape[0]])
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            plt.tight_layout()
            vizlabel = eval_labels[eval_dim] if eval_dim < len(eval_labels) else "demoviz"
            plt.savefig(base_path + 'viz_' + vizlabel + '_checkpoint' + str(save_idx) + '.png', dpi=300)
            plt.close()

            num_unique_values = len(unique_values)

            # Calculate the size of the subplot
            subplot_height = 0.9  # Height of the subplot in inches
            subplot_width = 0.6  # Width of each square plus spacing
            plt.figure(figsize=(subplot_width * num_unique_values, subplot_height))

            main_colormap = plt.get_cmap('hot')
            # Iterate through unique values and create colored squares
            spacing = 0.1  # Adjust the spacing between subimages
            text_y_offset = -0.5  # Adjust the vertical position of the text
            for i, value in enumerate(unique_values):
                color = main_colormap((i - min_val) / (max_val - min_val))
                x_position = i * subplot_width + 0.5
                plt.fill_between([x_position - 0.5 * subplot_width, x_position + 0.5 * subplot_width], 0, 1, color=color)
                plt.text(x_position, text_y_offset, f'{np.round(value, 2)}', color='black', ha='center', va='center')
            plt.ylim([text_y_offset - 0.1, 1])  # Adjust the y limits to include text
            plt.xlim([0, num_unique_values * (subplot_width + spacing)])
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            plt.savefig(base_path + 'viz_' + vizlabel + '_checkpoint' + str(save_idx) + '_swatches.png', dpi=300)
            plt.close()

        # And save the w_m matrices just as a backup in case we need them in the future
        with open(base_path + 'enc' + str(save_idx), 'wb') as f:
            pkl.dump(enc, f)

        overall_metrics = np.hstack([avg_colors, np.hstack(avg_rews)])
        abs_data = pd.DataFrame(overall_metrics, columns=['R', 'G', 'B', 'RewardR', 'RewardG', 'RewardB'])
        abs_data = abs_data.drop_duplicates()
        abs_data.to_csv(base_path + 'abstractions' + str(save_idx) + '.csv')

        distortion_data.append(dist_for_checkpoint)
        save_idx += 1
    plt.plot(I_x, I_y)
    plt.savefig(base_path + '/rew_IB.png')
    plt.close()

    comps = np.array(comps)
    comps = np.reshape(comps, (len(comps), 1))
    distortion_data = np.array(distortion_data)
    overall_metrics = np.hstack([comps, distortion_data])
    abs_data = pd.DataFrame(overall_metrics, columns=['Complexity'] + eval_labels)
    abs_data.to_csv(base_path + 'ib_results.csv')


def run():
    if demo:
        labels = ['demo_red_disco']
        rew_fns = [get_red_demo]
    else:
        labels = ['red_cont', 'green_cont', 'blue_cont']
        rew_fns = [get_red_cont, get_green_cont, get_blue_cont]
    for label, rew_fn in zip(labels, rew_fns):
        print("Running IB for training signal", label)

        run_dir = 'saved_data/exact_ib_rgb/train_' + label
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        run_for_setup(label, rew_fn)


def _get_disco(feature_idx):
    rews = [0.5, -0.5, 0., 0.75, 1.0, -1.0, 0.25, -0.75]
    rew_var = 0.2
    rew_matrix = np.zeros((330, len(rews)))
    for c1 in range(1, 331):
        features = raw_data.get_features(c1)
        feature = features[feature_idx]
        true_rew = _color_rew_fn(feature)
        # Get the distribution over rewards. Assume it's a Gaussian centered at the right point with fixed variance.
        rew_dist = np.array([np.exp(-1 * ((true_rew - r) ** 2) / rew_var) for r in rews])
        rew_dist = rew_dist / np.sum(rew_dist)
        rew_matrix[c1 - 1] = rew_dist
    return rew_matrix


def _get_cont(feature_idx):
    rew_var = 0.01
    all_vals = [raw_data.get_features(idx)[feature_idx] for idx in range(330)]
    rew_matrix = np.zeros((330, 330))
    for c1 in range(1, 331):
        feature = raw_data.get_features(c1)[feature_idx]
        # Get the distribution over possible other rgb values.
        # Assume it's a Gaussian centered at the right point with fixed variance.
        rew_dist = np.array([np.exp(-1 * ((feature - val) ** 2) / rew_var) for val in all_vals])
        rew_dist = rew_dist / np.sum(rew_dist)
        rew_matrix[c1 - 1] = rew_dist
    return rew_matrix


def _get_cid_to_rew(feature_idx):
    use_disco_fn = feature_idx < 3
    feature_idx = feature_idx % 3
    rew_matrix = np.zeros((330, 1))
    for c1 in range(1, 331):
        feature = raw_data.get_features(c1)[feature_idx]
        if use_disco_fn:
            feature = _color_rew_fn(feature)
        rew_matrix[c1 - 1] = feature
    return rew_matrix


def get_red_disco():
    return _get_disco(0)


def get_green_disco():
    return _get_disco(1)


def get_blue_disco():
    return _get_disco(2)


def get_red_demo():
    feature_idx = 0
    rews = [-0.5, -1.0, 1.0, 0.5]
    rew_var = 0.2
    rew_matrix = np.zeros((330, len(rews)))
    for c1 in range(1, 331):
        features = raw_data.get_features(c1)
        feature = features[feature_idx]
        if feature < 0.25:
            true_rew = rews[0]
        elif feature < 0.5:
            true_rew = rews[1]
        elif feature < 0.75:
            true_rew = rews[2]
        else:
            true_rew = rews[3]
        # Get the distribution over rewards. Assume it's a Gaussian centered at the right point with fixed variance.
        rew_dist = np.array([np.exp(-1 * ((true_rew - r) ** 2) / rew_var) for r in rews])
        rew_dist = rew_dist / np.sum(rew_dist)
        rew_matrix[c1 - 1] = rew_dist
    return rew_matrix


def get_red_demo_matrix():
    feature_idx = 0
    rews = [-0.5, -1.0, 1.0, 0.5]
    rew_matrix = np.zeros((330, 1))
    for c1 in range(1, 331):
        features = raw_data.get_features(c1)
        feature = features[feature_idx]
        if feature < 0.25:
            true_rew = rews[0]
        elif feature < 0.5:
            true_rew = rews[1]
        elif feature < 0.75:
            true_rew = rews[2]
        else:
            true_rew = rews[3]
        rew_matrix[c1 - 1] = true_rew
    return rew_matrix


def get_red_cont():
    return _get_cont(0)


def get_green_cont():
    return _get_cont(1)


def get_blue_cont():
    return _get_cont(2)


def _color_rew_fn(val):
    if val < 0.125:
        return 0.5
    if val < 0.25:
        return -0.5
    if val < 0.375:
        return 0
    if val < 0.5:
        return 0.75
    if val < 0.625:
        return 1.0
    if val < 0.75:
        return -1.0
    if val < 0.875:
        return 0.25
    return -0.75


if __name__ == '__main__':
    print("Doing exact IB")
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    demo = False
    ib_model = ib_naming_model.load_model()
    prior = ib_model.pM[:, 0]
    prior = np.ones_like(prior) / 330
    ib_model.pM[:, 0] = prior
    raw_data = WCSDataset()
    settings.color_space = 'rgb'
    run()
