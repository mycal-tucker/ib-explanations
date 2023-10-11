import os
import pickle as pkl
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from embo import InformationBottleneck


def _viz_grid(grid, path):
    min_val = np.min(grid) - 0.1
    max_val = np.max(grid) + 0.1
    plt.figure(figsize=(4, 4))
    im = plt.imshow(grid, cmap='hot', vmin=min_val, vmax=max_val)
    plt.colorbar(im, shrink=0.75)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    # Overlay white gridlines at every pixel
    for i in range(grid.shape[0]):
        plt.axhline(y=i - 0.5, color='white', linewidth=0.5)
    for j in range(grid.shape[1]):
        plt.axvline(x=j - 0.5, color='white', linewidth=0.5)
    plt.savefig(path, dpi=300)
    plt.close()

    return min_val, max_val


def run_for_setup(grid, rew_fn, label, base_path):
    rew_matrix = rew_fn(grid)
    ib = InformationBottleneck(pxy=rew_matrix, alpha=0, numbeta=30, restarts=10)
    I_x, I_y, H_m, beta = ib.get_bottleneck()
    abs_marg, encs, decs = ib.get_encs_decs()
    # Now just visualize the abstractions
    prev_comp = None
    step_size = 0.0001
    save_idx = 0
    distortion_data = []
    comps = []
    infos = []
    plt.plot(I_x, I_y)
    plt.savefig(base_path + '/rew_IB.png')
    plt.close()

    # Plot the real reward, just for future debugging.
    _viz_grid(grid, base_path + 'ground_truth.png')

    for idx, data in enumerate(zip(encs, decs, I_x, I_y)):
        enc, _, comp, info = data
        if prev_comp is not None and comp - prev_comp < step_size and idx != len(encs):
            continue
        prev_comp = comp
        comps.append(comp)
        infos.append(info)

        # Now generate visualizations according to the actual reward function, as well as x and y coordinates?
        viz_matrices = [get_rew_matrix_val(grid)]
        viz_labels = ['grid_value']
        dist_for_checkpoint = []
        print("Comp", comp)
        for viz_matrix, vizlabel in zip(viz_matrices, viz_labels):
            qMW = np.transpose(enc)
            pC = enc.sum(axis=0)[:, None]
            pW_C = np.where(pC > 0, qMW / pC, 1 / qMW.shape[1])
            y = pW_C.argmax(axis=1)
            pW = qMW.sum(axis=0)[:, None]
            pC_W = qMW.T / (pW + 1e-20)
            mu_w = pC_W.dot(viz_matrix)  # Reward associated with each abstraction
            modal_viz = mu_w[y]
            modal_viz = np.reshape(modal_viz, grid.shape)
            min_val, max_val = _viz_grid(modal_viz, base_path + 'viz_' + vizlabel + '_checkpoint' + str(save_idx) + '.png')

            # Generate the swatches as well.
            unique_vals = np.unique(modal_viz)
            num_unique_values = len(unique_vals)
            subplot_height = 0.9
            subplot_width = 0.6
            plt.figure(figsize=(subplot_width * num_unique_values, subplot_height))
            main_colormap = plt.get_cmap('hot')  # Iterate through unique values and create colored squares
            spacing = 0.1  # Adjust the spacing between subimages
            text_y_offset = -0.5  # Adjust the vertical position of the text
            for i, value in enumerate(unique_vals):
                color = main_colormap((value - min_val) / (max_val - min_val))
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

            task_distortion = np.mean((viz_matrix - mu_w[y]) ** 2)
            dist_for_checkpoint.append(task_distortion)
        distortion_data.append(dist_for_checkpoint)
        save_idx += 1

    comps = np.array(comps)
    comps = np.reshape(comps, (len(comps), 1))
    distortion_data = np.array(distortion_data)
    overall_metrics = np.hstack([comps, distortion_data])
    abs_data = pd.DataFrame(overall_metrics, columns=['Complexity', 'Reward Distortion'])
    abs_data.to_csv(base_path + 'ib_results.csv')


def run():
    # We define a large number of grids.
    # Simple ones are useful for debugging; more complex ones were used in the paper submission.
    grids = [np.array([[-0.5, 0.1, 0.2],
                     [-0.5, 0.0, -0.5]]),
             np.array([[-0.5, 0.1, 0.2, 0.2, 0.0],
                       [-0.5, 0.0, 0.2, 0.0, 0.0],
                       [-0.5, 0.0, 0.0, 1.0, 1.0],
                       [0.0, 0.0, -1.0, 1.0, 1.0],
                       [0.0, 0.0, 0.0, -0.5, -0.5]]),
             np.array([[-1.0, -0.5, 0.0, 0.5, 1.0] for _ in range(5)]),
             np.array([[-1.0 + i * 0.5 for _ in range(5)] for i in range(5)])]
    manhattan_grid = np.zeros((5, 5))
    goal = (1, 3)
    max_dist = 0
    for i in range(5):
        for j in range(5):
            dist = np.abs(j - goal[0]) + np.abs(i - goal[1])
            manhattan_grid[j, i] = dist
            max_dist = max([dist, max_dist])
    manhattan_grid = 2 * manhattan_grid / max_dist
    manhattan_grid = 1 - manhattan_grid
    grids.append(manhattan_grid)
    for _ in range(3):  # Generate some random grids
        rand_grid = np.random.random((5, 5)) * 2 - 1
        grids.append(rand_grid)
    grids.append(np.array([[1, 1, 0.5, 0.5, 0.5],
              [1, 1, 0.5, 0.5, 0.5],
              [0, 0, 0, 0, 0],
              [-0.5, -0.5, -1, -1, -1],
              [-0.5, -0.5, -1, -1, -1]]))
    labels = ['x_val', 'y_val', 'rew_val']
    rew_fns = [rew_fn_x, rew_fn_y, rew_fn_val]
    for i, grid in enumerate(grids):
        print("Grid\n", grid)
        for label, rew_fn in zip(labels, rew_fns):
            run_dir = 'saved_data/exact_ib_grid/grid' + str(i) + '/train_' + label + '/'
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)
            with open(run_dir + 'grid.pkl', 'wb') as f:
                pkl.dump(grid, f)
            run_for_setup(grid, rew_fn, label, run_dir)


def rew_fn_x(grid):
    rew_matrix = np.zeros((np.product(grid.shape), grid.shape[1]))
    dim_var = 0.5
    rews = [i for i in range(grid.shape[1])]
    idx = 0
    for row in grid:
        for x, val in enumerate(row):
            # Get a distribution over x values
            rew_dist = np.array([np.exp(-1 * ((x - r) ** 2) / dim_var) for r in rews])
            rew_dist = rew_dist / np.sum(rew_dist)
            rew_matrix[idx] = rew_dist
            idx += 1
    return rew_matrix


def rew_fn_y(grid):
    rew_matrix = np.zeros((np.product(grid.shape), grid.shape[0]))
    dim_var = 0.5
    rews = [i for i in range(grid.shape[0])]
    idx = 0
    for y, row in enumerate(grid):
        for _ in row:
            # Get a distribution over x values
            rew_dist = np.array([np.exp(-1 * ((y - r) ** 2) / dim_var) for r in rews])
            rew_dist = rew_dist / np.sum(rew_dist)
            rew_matrix[idx] = rew_dist
            idx += 1
    return rew_matrix


def rew_fn_val(grid):
    unique_vals = np.unique(grid)
    rew_matrix = np.zeros((np.product(grid.shape), len(unique_vals)))
    rew_var = 0.2
    rews = [v for v in unique_vals]
    idx = 0
    for row in grid:
        for val in row:
            # Get a distribution over x values
            rew_dist = np.array([np.exp(-1 * ((val - r) ** 2) / rew_var) for r in rews])
            rew_dist = rew_dist / np.sum(rew_dist)
            rew_matrix[idx] = rew_dist
            idx += 1
    return rew_matrix


def get_rew_matrix_val(grid):
    rews = []
    for row in grid:
        for val in row:
            rews.append(val)
    return np.array(rews)


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    run()
