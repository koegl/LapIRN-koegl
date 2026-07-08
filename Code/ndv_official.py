import numpy as np


def calc_measurements(jac_dets, mask):

    # calculate non-diffeomorphic voxels
    non_diff_voxels = np.sum((jac_dets["0x0y0z"] <= 0) * mask)

    # calculate non-diffeomorphic voxels
    non_diff_tetrahedra = 0
    weights = [
        1 / 5 / 2
    ] * 10  # equally weighted, five tetrahedra, two dividing schemes
    # weights = [1/6/2] * 8 + [1/3/2] * 2 # weighted based on volume
    for w, grad_args in zip(
        weights,
        [
            "+x+y+z",
            "+x+y-z",
            "+x-y+z",
            "+x-y-z",
            "-x+y+z",
            "-x+y-z",
            "-x-y+z",
            "-x-y-z",
            "Jstar_1",
            "Jstar_2",
        ],
    ):
        non_diff_tetrahedra += np.sum((jac_dets[grad_args] <= 0) * mask) * w

    # calculate non-diffeomorphic volume
    non_diff_volume = 0
    non_diff_volume_map = 0
    for grad_args in [
        "+x+y+z",
        "+x+y-z",
        "+x-y+z",
        "+x-y-z",
        "Jstar_1",
        "Jstar_2",
        "-x+y+z",
        "-x+y-z",
        "-x-y+z",
        "-x-y-z",
    ]:
        # each |J| equals six times the signed volume, two schemes considered
        non_diff_volume += np.sum(-0.5 * np.minimum(jac_dets[grad_args], 0) * mask / 6)
        non_diff_volume_map += -0.5 * np.minimum(jac_dets[grad_args], 0) * mask / 6

    return non_diff_voxels, non_diff_tetrahedra, non_diff_volume, non_diff_volume_map


import scipy.ndimage


def calc_J_i(trans, grad_args):
    kernel = {}
    grad_args = list(grad_args)
    grad_args = [(grad_args[i] + grad_args[i + 1]) for i in range(0, len(grad_args), 2)]

    # example: [('-', 'x'), ('+', 'y'), ('-', 'z')]
    for direction, axis in grad_args:
        if direction == "+":
            kernel[axis] = np.array([0, -1, 1])
        elif direction == "0":
            kernel[axis] = np.array([-0.5, 0, 0.5])
        elif direction == "-":
            kernel[axis] = np.array([-1, 1, 0])
    kernel["x"] = kernel["x"].reshape(1, 3, 1, 1)
    kernel["y"] = kernel["y"].reshape(1, 1, 3, 1)
    kernel["z"] = kernel["z"].reshape(1, 1, 1, 3)

    trans = trans[None, ...]
    gradx = np.stack(
        [
            scipy.ndimage.correlate(trans[:, 0, :, :, :], kernel["x"], mode="nearest"),
            scipy.ndimage.correlate(trans[:, 1, :, :, :], kernel["x"], mode="nearest"),
            scipy.ndimage.correlate(trans[:, 2, :, :, :], kernel["x"], mode="nearest"),
        ],
        axis=1,
    )

    grady = np.stack(
        [
            scipy.ndimage.correlate(trans[:, 0, :, :, :], kernel["y"], mode="nearest"),
            scipy.ndimage.correlate(trans[:, 1, :, :, :], kernel["y"], mode="nearest"),
            scipy.ndimage.correlate(trans[:, 2, :, :, :], kernel["y"], mode="nearest"),
        ],
        axis=1,
    )

    gradz = np.stack(
        [
            scipy.ndimage.correlate(trans[:, 0, :, :, :], kernel["z"], mode="nearest"),
            scipy.ndimage.correlate(trans[:, 1, :, :, :], kernel["z"], mode="nearest"),
            scipy.ndimage.correlate(trans[:, 2, :, :, :], kernel["z"], mode="nearest"),
        ],
        axis=1,
    )

    jacobian = np.concatenate([gradx, grady, gradz], 0)

    jac_det = (
        jacobian[0, 0, :, :, :]
        * (
            jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :]
            - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]
        )
        - jacobian[1, 0, :, :, :]
        * (
            jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :]
            - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]
        )
        + jacobian[2, 0, :, :, :]
        * (
            jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :]
            - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :]
        )
    )

    jac_det = jac_det[1:-1, 1:-1, 1:-1]  # remove boundary voxels

    return jac_det


def calc_Jstar_1(trans):
    kernel = {}
    kernel["x"] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]).reshape(1, 3, 3, 1)
    kernel["y"] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]).reshape(1, 3, 1, 3)
    kernel["z"] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]).reshape(1, 1, 3, 3)

    trans = trans[None, ...]
    gradx = np.stack(
        [
            scipy.ndimage.correlate(trans[:, 0, :, :, :], kernel["x"], mode="nearest"),
            scipy.ndimage.correlate(trans[:, 1, :, :, :], kernel["x"], mode="nearest"),
            scipy.ndimage.correlate(trans[:, 2, :, :, :], kernel["x"], mode="nearest"),
        ],
        axis=1,
    )

    grady = np.stack(
        [
            scipy.ndimage.correlate(trans[:, 0, :, :, :], kernel["y"], mode="nearest"),
            scipy.ndimage.correlate(trans[:, 1, :, :, :], kernel["y"], mode="nearest"),
            scipy.ndimage.correlate(trans[:, 2, :, :, :], kernel["y"], mode="nearest"),
        ],
        axis=1,
    )

    gradz = np.stack(
        [
            scipy.ndimage.correlate(trans[:, 0, :, :, :], kernel["z"], mode="nearest"),
            scipy.ndimage.correlate(trans[:, 1, :, :, :], kernel["z"], mode="nearest"),
            scipy.ndimage.correlate(trans[:, 2, :, :, :], kernel["z"], mode="nearest"),
        ],
        axis=1,
    )

    jacobian = np.concatenate([gradx, grady, gradz], 0)

    jac_det = (
        jacobian[0, 0, :, :, :]
        * (
            jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :]
            - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]
        )
        - jacobian[1, 0, :, :, :]
        * (
            jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :]
            - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]
        )
        + jacobian[2, 0, :, :, :]
        * (
            jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :]
            - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :]
        )
    )

    jac_det = jac_det[1:-1, 1:-1, 1:-1]  # remove boundary voxels

    return jac_det


def calc_Jstar_2(trans):
    kernel = {}
    kernel["x"] = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]]).reshape(1, 3, 3, 1)
    kernel["y"] = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]]).reshape(1, 1, 3, 3)
    kernel["z"] = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]]).reshape(1, 3, 1, 3)

    trans = trans[None, ...]
    gradx = np.stack(
        [
            scipy.ndimage.correlate(trans[:, 0, :, :, :], kernel["x"], mode="nearest"),
            scipy.ndimage.correlate(trans[:, 1, :, :, :], kernel["x"], mode="nearest"),
            scipy.ndimage.correlate(trans[:, 2, :, :, :], kernel["x"], mode="nearest"),
        ],
        axis=1,
    )

    grady = np.stack(
        [
            scipy.ndimage.correlate(trans[:, 0, :, :, :], kernel["y"], mode="nearest"),
            scipy.ndimage.correlate(trans[:, 1, :, :, :], kernel["y"], mode="nearest"),
            scipy.ndimage.correlate(trans[:, 2, :, :, :], kernel["y"], mode="nearest"),
        ],
        axis=1,
    )

    gradz = np.stack(
        [
            scipy.ndimage.correlate(trans[:, 0, :, :, :], kernel["z"], mode="nearest"),
            scipy.ndimage.correlate(trans[:, 1, :, :, :], kernel["z"], mode="nearest"),
            scipy.ndimage.correlate(trans[:, 2, :, :, :], kernel["z"], mode="nearest"),
        ],
        axis=1,
    )

    jacobian = np.concatenate([gradx, grady, gradz], 0)

    jac_det = (
        jacobian[0, 0, :, :, :]
        * (
            jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :]
            - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]
        )
        - jacobian[1, 0, :, :, :]
        * (
            jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :]
            - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]
        )
        + jacobian[2, 0, :, :, :]
        * (
            jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :]
            - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :]
        )
    )

    jac_det = jac_det[1:-1, 1:-1, 1:-1]  # remove boundary voxels

    return jac_det


def calc_jac_dets(trans):
    jac_det = {}
    # calculate all finite difference based |J|'s
    for grad_args in [
        "0x0y0z",
        "+x+y+z",
        "+x+y-z",
        "+x-y+z",
        "+x-y-z",
        "-x+y+z",
        "-x+y-z",
        "-x-y+z",
        "-x-y-z",
    ]:
        jac_det[grad_args] = calc_J_i(trans, grad_args)

    # calc 'all |J_i| > 0'
    jac_det["all J_i>0"] = np.ones_like(jac_det["0x0y0z"])
    for grad_args in [
        "+x+y+z",
        "+x+y-z",
        "+x-y+z",
        "+x-y-z",
        "-x+y+z",
        "-x+y-z",
        "-x-y+z",
        "-x-y-z",
    ]:
        jac_det["all J_i>0"] *= jac_det[grad_args] > 0

    # sanity check: if 'all J_i > 0', the central difference must be positive
    assert np.sum((jac_det["all J_i>0"] > 0) * (jac_det["0x0y0z"] <= 0)) == 0

    jac_det["Jstar_1"] = calc_Jstar_1(trans)
    jac_det["Jstar_2"] = calc_Jstar_2(trans)

    return jac_det


def get_identity_grid(array):
    """Return the identity transformation of the same size as the input.
    Expect input dimension: 3xHxWxS."""
    dims = array.shape[1:]
    vectors = [np.arange(0, dim, 1) for dim in dims]

    grids = np.meshgrid(*vectors)
    grids = [np.transpose(x, axes=(1, 0, 2)) for x in grids]
    grid = np.stack(grids).astype("float32")

    return grid
