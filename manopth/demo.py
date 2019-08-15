import numpy as np
import torch
from manopth.manolayer import ManoLayer
from matplotlib import pyplot as plt  # 把画图放在最后面
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def generate_random_hand(batch_size=1, ncomps=6, mano_root='mano/models'):
    nfull_comps = ncomps + 3  # Add global orientation dims to PCA
    random_pcapose = torch.rand(batch_size, nfull_comps)
    mano_layer = ManoLayer(mano_root=mano_root)
    verts, joints = mano_layer(random_pcapose)
    return {'verts': verts, 'joints': joints, 'faces': mano_layer.th_faces}


def display_hand(hand_info, mano_faces=None, ax=None, alpha=0.2, batch_idx=0, show=True):
    """
    Displays hand batch_idx in batch of hand_info, hand_info as returned by
    generate_random_hand
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    verts, joints = hand_info['verts'][batch_idx], hand_info['joints'][batch_idx]
    if mano_faces is None:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.1)
    else:
        mesh = Poly3DCollection(verts[mano_faces], alpha=alpha)
        face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
        # pass
    joints = joints.numpy()
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='y')
    ax.plot(joints[[0, 1, 2, 3, 4], 0], joints[[0, 1, 2, 3, 4], 1], joints[[0, 1, 2, 3, 4], 2], color='r')
    ax.plot(joints[[0, 5, 6, 7, 8], 0], joints[[0, 5, 6, 7, 8], 1], joints[[0, 5, 6, 7, 8], 2], color='r')
    ax.plot(joints[[0, 9, 10, 11, 12], 0], joints[[0, 9, 10, 11, 12], 1], joints[[0, 9, 10, 11, 12], 2], color='r')
    ax.plot(joints[[0, 13, 14, 15, 16], 0], joints[[0, 13, 14, 15, 16], 1], joints[[0, 13, 14, 15, 16], 2], color='r')
    ax.plot(joints[[0, 17, 18, 19, 20], 0], joints[[0, 17, 18, 19, 20], 1], joints[[0, 17, 18, 19, 20], 2], color='r')
    # cam_equal_aspect_3d(ax, verts)
    if show:
        plt.show()


def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)
