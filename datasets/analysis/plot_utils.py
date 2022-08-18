import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import matplotlib.patheffects as pe


def rotmat2d(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def rotate(a, angle):
    return np.matmul(rotmat2d(angle), a.T).T


def plot_2d_trajectory(positions, labels=None, label_steps=25, label_radius=30., label_angle=135., title=None,
                       save_path=None):

    positions = positions[:, :2]

    if labels is None:
        labels = np.arange(len(positions))

    indices = np.arange(len(positions))

    labels_mask = indices % label_steps == 0
    labels = labels[labels_mask]

    txt_poses = positions[labels_mask]

    # Compute the gradient of the trajectory to know where to put the labels
    pdiff = np.diff(positions, axis=0)
    pgrad_norm = np.sqrt(np.sum(np.square(pdiff), axis=1))
    pgrad_norm[pgrad_norm == 0] = 1.0  # To avoid division by zero
    pgrad_norm = pgrad_norm.reshape((-1, 1))
    pgrad = pdiff / pgrad_norm  # Normalize the gradient to a unit-length vector
    pgrad = np.vstack([pgrad[[0]], pgrad])  # So that the first point has the same gradient as the next one
    pgrad = pgrad[labels_mask]  # Slice it so that only the gradient for the points of interest remain

    min_lim = np.min(positions, axis=0)
    max_lim = np.max(positions, axis=0)

    # Convert angle to radians
    label_angle = label_angle * np.pi / 180.

    tmp_pos_inc = rotate(pgrad, label_angle)

    # Code for modifying the labels that happen to fall close or on top of other labels
    txt_pos = txt_poses + label_radius * tmp_pos_inc

    pgrad_grad_x = np.square(txt_pos[:, 0] - txt_pos[:, 0, np.newaxis])
    pgrad_grad_y = np.square(txt_pos[:, 1] - txt_pos[:, 1, np.newaxis])
    pgrad_grad = np.sqrt(pgrad_grad_x + pgrad_grad_y)
    pgrad_grad = np.tril(pgrad_grad)
    pgrad_grad = pgrad_grad < (label_radius / 2)
    pgrad_grad = np.logical_and(pgrad_grad, np.tril(np.ones_like(pgrad_grad)))
    pgrad_grad = np.logical_and(pgrad_grad, np.logical_not(np.diag(np.ones_like(pgrad_grad[0]))))
    pgrad_grad_counts = np.sum(pgrad_grad, axis=1)

    inv_th_msk = pgrad_grad_counts > 0
    txt_pos[inv_th_msk] = txt_poses[inv_th_msk] + label_radius * rotate(pgrad[inv_th_msk], -label_angle)

    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.set_axisbelow(True)
    plt.grid(which="minor", axis="both", color="black", linestyle="--", linewidth=0.25)
    plt.grid(which="major", axis="both", color="black", linestyle="--", linewidth=0.375)

    ax.scatter(positions[0, 0], positions[0, 1],
               s=8, c="black", marker="^", linewidths=0.0)
    ax.scatter(positions[-1, 0], positions[-1, 1],
               s=8, c="black", marker="*", linewidths=0.0)
    square_mask = labels_mask
    square_mask[0] = False
    square_mask[-1] = False
    ax.scatter(positions[square_mask, 0], positions[square_mask, 1],
               s=2.5, c="black", marker="s", linewidths=0.0)
    circ_mask = np.logical_not(square_mask)
    circ_mask[0] = False
    circ_mask[-1] = False
    ax.scatter(positions[circ_mask, 0], positions[circ_mask, 1],
               s=0.875, c="black", marker="o", linewidths=0.0)

    for label, point, label_pos in zip(labels, txt_poses, txt_pos):
        ax.annotate(label, point, xytext=label_pos, fontsize=7,
                    arrowprops={'arrowstyle': "->",
                                'linewidth': 0.5
                                }
                    )
    min_lim -= 1.5 * label_radius
    max_lim += 1.5 * label_radius
    ax.set_aspect("equal")
    ax.set_xlim([min_lim[0], max_lim[0]])
    ax.set_ylim([min_lim[1], max_lim[1]])
    if title:
        ax.set_title(title)
    ax.set_xlabel("x\n[m]")
    ax.set_ylabel("z\n[m]", rotation=0)

    if save_path:
        plt.savefig(save_path)
        print(f"Trajectory figure saved in {save_path.stem}.")

    plt.close(fig)


def plot_3d_trajectory(positions, save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax.xaxis._axinfo["grid"].update({"linewidth": 0.25, "color": "k"})
    ax.yaxis._axinfo["grid"].update({"linewidth": 0.25, "color": "k"})
    ax.zaxis._axinfo["grid"].update({"linewidth": 0.25, "color": "k"})

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    lw = 0.75
    ax.plot3D(positions[:, 0], positions[:, 1], positions[:, 2], c="k", lw=0.75,
              path_effects=[pe.Stroke(linewidth=3 * lw, foreground="white"), pe.Normal()])

    plt.grid(which="minor", axis="both", color="black", linestyle="--", linewidth=0.25)
    plt.grid(which="major", axis="both", color="black", linestyle="--", linewidth=0.375)

    if save_path:
        plt.savefig(save_path)
        print(f"Trajectory figure saved in {save_path.stem}.")

    plt.close(fig)


def plot_2_5d_trajectory(positions, z_step=1, save_path=None):

    n_poses = positions.shape[0]
    positions_2_5d = np.zeros((n_poses, 3))
    positions_2_5d[:, :2] = positions[:, :2]
    positions_2_5d[:, 2] = np.arange(n_poses) * z_step

    plot_3d_trajectory(positions_2_5d, save_path=save_path)


def plot_pointcloud(pointcloud):
    fig = plt.figure(figsize=(6, 8), dpi=500)
    ax = fig.add_subplot(projection='3d')
    # plt.gca().set_aspect('equal')
    ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], marker="o", c="k", s=0.01)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()
