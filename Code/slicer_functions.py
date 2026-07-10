from pathlib import Path
from typing import Sequence

import numpy as np
import slicer


def load_bone_labels(seg_path: str):
    """Load a segmentation, merge all bone labels into one red outline segment.

    Args:
        seg_path: Path to the segmentation NIfTI (labelmap).

    Returns:
        The created segmentation node.
    """
    bone_label_values: Sequence[int] = (
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
    )

    node_name = Path(seg_path).name.replace(".nii.gz", "").replace(".nii", "")
    node_name += "_bone"

    labelmap_node = slicer.util.loadLabelVolume(seg_path)

    arr = slicer.util.arrayFromVolume(labelmap_node)
    keep = np.isin(arr, bone_label_values)
    arr[keep] = 1
    arr[~keep] = 0
    slicer.util.updateVolumeFromArray(labelmap_node, arr)

    seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", node_name)
    seg_node.CreateDefaultDisplayNodes()
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
        labelmap_node, seg_node
    )
    slicer.mrmlScene.RemoveNode(labelmap_node)

    segmentation = seg_node.GetSegmentation()
    segment_id = segmentation.GetNthSegmentID(0)
    segmentation.GetSegment(segment_id).SetColor(1.0, 0.0, 0.0)

    display_node = seg_node.GetDisplayNode()
    display_node.SetOpacity2DFill(0.0)
    display_node.SetOpacity2DOutline(1.0)

    return seg_node
