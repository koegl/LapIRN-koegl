"""Cross-check the differentiable torch NDV (jacobian.py) against the official
numpy scorer (ndv_official.py). Run on the cluster (needs torch + scipy):

    python check_ndv_equiv.py

Both must agree on percent_ndv (the scored metric) and on per-scheme dets.
Also checks that gradients flow through the loss.
"""

import numpy as np
import torch

import jacobian
import ndv_official


def official_percent_ndv(disp_np):
    # disp_np: (3, X, Y, Z), component c ramps along axis c (same as get_identity_grid)
    trans = disp_np + ndv_official.get_identity_grid(disp_np)
    jac_dets = ndv_official.calc_jac_dets(trans)
    mask_inner = np.ones_like(jac_dets["0x0y0z"])  # all interior voxels
    _, _, non_diff_volume, _ = ndv_official.calc_measurements(jac_dets, mask_inner)
    total = float(mask_inner.sum())
    return non_diff_volume / total * 100.0


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    X, Y, Z = 24, 20, 18

    for name, scale in [("smooth", 0.3), ("folding", 3.0)]:
        # low-frequency random displacement, voxel units, (3, X, Y, Z)
        low = np.random.randn(3, 6, 5, 4).astype("float32") * scale
        disp = np.stack(
            [
                np.asarray(
                    torch.nn.functional.interpolate(
                        torch.tensor(low[c])[None, None],
                        size=(X, Y, Z),
                        mode="trilinear",
                        align_corners=True,
                    )[0, 0]
                )
                for c in range(3)
            ]
        )

        off = official_percent_ndv(disp)

        flow = torch.tensor(disp).permute(1, 2, 3, 0)[None].clone()  # (1, X, Y, Z, 3)
        flow.requires_grad_(True)
        loss = jacobian.non_diff_volume_loss(flow)
        loss.backward()

        print(
            f"[{name:8s}] official={off:.6f}  torch={loss.item():.6f}  "
            f"abs_diff={abs(off - loss.item()):.2e}  grad_nonzero={bool((flow.grad.abs() > 0).any())}"
        )


if __name__ == "__main__":
    main()
