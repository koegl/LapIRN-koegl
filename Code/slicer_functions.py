def load_bone_labels(seg_path: str):
    """Load a segmentation, merge all bone labels into one red outline segment.

    Args:
        seg_path: Path to the segmentation NIfTI (labelmap).

    Returns:
        The created segmentation node.
    """
    from pathlib import Path
    from typing import Sequence

    import numpy as np
    import slicer

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


def load_rib_labels(seg_path: str):
    """Load a segmentation, merge all bone labels into one red outline segment.

    Args:
        seg_path: Path to the segmentation NIfTI (labelmap).

    Returns:
        The created segmentation node.
    """
    from pathlib import Path
    from typing import Sequence

    import numpy as np
    import slicer
    import vtk

    rib_label_values: Sequence[int] = (
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
    )

    node_name = Path(seg_path).name.replace(".nii.gz", "").replace(".nii", "")

    labelmap_node = slicer.util.loadLabelVolume(seg_path)

    arr = slicer.util.arrayFromVolume(labelmap_node)
    keep = np.isin(arr, rib_label_values)
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

    sh_node = slicer.mrmlScene.GetSubjectHierarchyNode()
    export_folder_id = sh_node.CreateFolderItem(
        sh_node.GetSceneItemID(), node_name + "_models"
    )
    slicer.modules.segmentations.logic().ExportVisibleSegmentsToModels(
        seg_node, export_folder_id
    )

    model_ids = vtk.vtkIdList()
    sh_node.GetItemChildren(export_folder_id, model_ids)
    for i in range(model_ids.GetNumberOfIds()):
        model_node = sh_node.GetItemDataNode(model_ids.GetId(i))
        if model_node is not None:
            slicer.mrmlScene.RemoveNode(model_node)
    sh_node.RemoveItem(export_folder_id)

    return seg_node


def load_abdomen_labels(seg_path: str):
    """Load a segmentation, merge all bone labels into one red outline segment.

    Args:
        seg_path: Path to the segmentation NIfTI (labelmap).

    Returns:
        The created segmentation node.
    """
    from pathlib import Path
    from typing import Sequence

    import numpy as np
    import slicer
    import vtk

    abdomen_label_values: Sequence[int] = (
        27,
        28,
        29,
        30,
        31,
        32,
        33,
    )

    node_name = Path(seg_path).name.replace(".nii.gz", "").replace(".nii", "")

    labelmap_node = slicer.util.loadLabelVolume(seg_path)

    arr = slicer.util.arrayFromVolume(labelmap_node)
    keep = np.isin(arr, abdomen_label_values)
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

    sh_node = slicer.mrmlScene.GetSubjectHierarchyNode()
    export_folder_id = sh_node.CreateFolderItem(
        sh_node.GetSceneItemID(), node_name + "_models"
    )
    slicer.modules.segmentations.logic().ExportVisibleSegmentsToModels(
        seg_node, export_folder_id
    )

    model_ids = vtk.vtkIdList()
    sh_node.GetItemChildren(export_folder_id, model_ids)
    for i in range(model_ids.GetNumberOfIds()):
        model_node = sh_node.GetItemDataNode(model_ids.GetId(i))
        if model_node is not None:
            slicer.mrmlScene.RemoveNode(model_node)
    sh_node.RemoveItem(export_folder_id)

    return seg_node


def load_nlst_labels(seg_path: str):
    """Load a segmentation, merge all bone labels into one red outline segment.

    Args:
        seg_path: Path to the segmentation NIfTI (labelmap).

    Returns:
        The created segmentation node.
    """
    from pathlib import Path
    from typing import Sequence

    import numpy as np
    import slicer
    import vtk

    abdomen_label_values: Sequence[int] = (
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
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
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
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
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
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
        117,
    )

    node_name = Path(seg_path).name.replace(".nii.gz", "").replace(".nii", "")

    labelmap_node = slicer.util.loadLabelVolume(seg_path)

    arr = slicer.util.arrayFromVolume(labelmap_node)
    keep = np.isin(arr, abdomen_label_values)
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

    sh_node = slicer.mrmlScene.GetSubjectHierarchyNode()
    export_folder_id = sh_node.CreateFolderItem(
        sh_node.GetSceneItemID(), node_name + "_models"
    )
    slicer.modules.segmentations.logic().ExportVisibleSegmentsToModels(
        seg_node, export_folder_id
    )

    model_ids = vtk.vtkIdList()
    sh_node.GetItemChildren(export_folder_id, model_ids)
    for i in range(model_ids.GetNumberOfIds()):
        model_node = sh_node.GetItemDataNode(model_ids.GetId(i))
        if model_node is not None:
            slicer.mrmlScene.RemoveNode(model_node)
    sh_node.RemoveItem(export_folder_id)

    return seg_node


def merge_all_segments(seg_node_name: str, node_name: str = None):
    """Combine all segments of a segmentation node into one, in a new node.

    Args:
        seg_node_name: Name of the source segmentation node.
        node_name: Name for the new node. Defaults to source name + '_merged'.

    Returns:
        The new segmentation node with a single merged segment.
    """
    import slicer

    src_node = slicer.util.getNode(seg_node_name)

    if node_name is None:
        node_name = seg_node_name + "_merged"

    labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
        src_node, labelmap
    )

    arr = slicer.util.arrayFromVolume(labelmap)
    arr[arr != 0] = 1
    slicer.util.updateVolumeFromArray(labelmap, arr)

    merged_node = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLSegmentationNode", node_name
    )
    merged_node.CreateDefaultDisplayNodes()
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
        labelmap, merged_node
    )
    slicer.mrmlScene.RemoveNode(labelmap)

    return merged_node
