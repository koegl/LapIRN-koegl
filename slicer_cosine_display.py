"""Load one voxelwise_cosine.py output folder into 3D Slicer with sensible display.

Run inside Slicer's Python console:

    folder = "/home/iml/fryderyk.koegl/code/LapIRN-koegl/saved/voxelwise_cosine/PSMARegPSMA_0006_02_to_00/best_step_069"
    exec(open("/home/iml/fryderyk.koegl/code/LapIRN-koegl/slicer_cosine_display.py").read())

Then switch the foreground layer between the loaded maps in the slice views.

Display per map type:
  *_cos, *_cosmag  signed: blue (disagree) - red (agree), inactive voxels fully
                   transparent so the warped moving CT shows through.
                   Slicer only makes voxels transparent through the threshold,
                   which keeps ONE contiguous range -- useless when 0 sits in the
                   middle of a signed map. So each signed map gets a display copy
                   (<name>_show) encoded as
                       0                                 inactive voxel
                       1.5 + 0.5 * clip(x / p99, -1, 1)  otherwise, in [1, 2]
                   thresholded to [1, 2] and coloured diverging over [1, 2].
                   Inactive: |cosmag| < HIDE_BELOW * its p99; for *_cos, the
                   pair's *_mag < HIDE_BELOW * its p99 (cos alone is +-1 on
                   numerically-zero gradients).
  *_mag, grad_*    unsigned: inferno from 0 to the 99th percentile, voxels
                   below 1% of that threshold are hidden.
Background: warped moving CT (window 400 / level 40); the warped tumour label
is loaded as a segmentation-style label outline for orientation.
"""

import glob
import os

import numpy as np
import slicer

PERCENTILE = 99.0
HIDE_BELOW = 0.05  # fraction of the p99 below which a voxel is transparent


def robust_max(node, signed):
    arr = slicer.util.arrayFromVolume(node)
    nz = arr[arr != 0]
    if nz.size == 0:
        return 1.0
    return float(np.percentile(np.abs(nz) if signed else nz, PERCENTILE))


def diverging_colormap():
    for name in ("DivergingBlueRed", "ColdToHotRainbow"):
        node = slicer.mrmlScene.GetFirstNodeByName(name)
        if node is not None:
            return node
    raise RuntimeError("no diverging colour table found")


def show_signed(node, mag_node=None):
    """Create <name>_show (see module docstring) and display it; returns it."""
    arr = slicer.util.arrayFromVolume(node).astype(np.float32)
    scale = 1.0 if node.GetName().endswith("_cos") else robust_max(node, signed=True)
    if mag_node is not None:
        mag = slicer.util.arrayFromVolume(mag_node)
        active = mag >= HIDE_BELOW * robust_max(mag_node, signed=False)
    else:
        active = np.abs(arr) >= HIDE_BELOW * scale
    shown = np.where(active, 1.5 + 0.5 * np.clip(arr / scale, -1.0, 1.0), 0.0)

    show = slicer.modules.volumes.logic().CloneVolume(
        slicer.mrmlScene, node, node.GetName() + "_show"
    )
    slicer.util.updateVolumeFromArray(show, shown.astype(np.float32))
    d = show.GetDisplayNode()
    d.SetAndObserveColorNodeID(diverging_colormap().GetID())
    d.SetInterpolate(0)
    d.AutoWindowLevelOff()
    d.SetWindowLevel(1.0, 1.5)
    d.AutoThresholdOff()
    d.SetLowerThreshold(1.0)
    d.SetUpperThreshold(2.0)
    d.ApplyThresholdOn()
    return show


def show_unsigned(node):
    m = robust_max(node, signed=False)
    d = node.GetDisplayNode()
    d.SetAndObserveColorNodeID(slicer.util.getNode("Inferno").GetID())
    d.AutoWindowLevelOff()
    d.SetWindowLevel(m, m / 2)
    d.AutoThresholdOff()
    d.SetLowerThreshold(0.01 * m)
    d.SetUpperThreshold(float(slicer.util.arrayFromVolume(node).max()))
    d.ApplyThresholdOn()


ct = tumour = first_map = None
nodes = {}
for path in sorted(glob.glob(os.path.join(folder, "*.nii.gz"))):  # noqa: F821
    name = os.path.basename(path).removesuffix(".nii.gz")
    if name.startswith("warped_moving_tumour") or name.startswith("warped_moving_organs"):
        label = slicer.util.loadLabelVolume(path, properties={"name": name, "show": False})
        if name.startswith("warped_moving_tumour"):
            tumour = label
        continue
    nodes[name] = slicer.util.loadVolume(path, properties={"name": name, "show": False})

for name, node in nodes.items():
    if name.startswith("warped_moving_ct"):
        ct = node
        node.GetDisplayNode().AutoWindowLevelOff()
        node.GetDisplayNode().SetWindowLevel(400, 40)
    elif name.startswith("warped_moving_pet"):
        node.GetDisplayNode().SetAndObserveColorNodeID(
            slicer.util.getNode("Inferno").GetID()
        )
    elif name.endswith("_cos"):
        show_signed(node, mag_node=nodes.get(name.removesuffix("_cos") + "_mag"))
    elif name.endswith("_cosmag"):
        shown = show_signed(node)
        if first_map is None:
            first_map = shown
    else:
        show_unsigned(node)

slicer.util.setSliceViewerLayers(
    background=ct, foreground=first_map, foregroundOpacity=1.0, label=tumour
)
for view in ("Red", "Yellow", "Green"):
    widget = slicer.app.layoutManager().sliceWidget(view)
    widget.sliceLogic().GetSliceCompositeNode().SetLabelOpacity(1.0)
    widget.mrmlSliceNode().SetUseLabelOutline(True)
