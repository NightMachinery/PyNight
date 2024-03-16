import numpy
import numpy as np
from PIL import Image
from pynight.common_icecream import ic
from pynight.common_iterable import to_iterable


##
def seg_id_to_mask_dict(
    seg_id,
    exclude_ids=None,
):
    exclude_ids = to_iterable(exclude_ids)

    unique_ids = np.unique(seg_id)
    mask_dict = {}
    for unique_id in unique_ids:
        if unique_id in exclude_ids:
            continue

        mask_dict[unique_id] = seg_id == unique_id
    return mask_dict


def overlay_masks_on_image(
    image_tensor,
    segmasks_dict,
    alpha=0.5,
    class_colors=None,
    input_dim_mode=None,
    input_range="255",
):
    if class_colors is None:
        class_colors = dict()

    image_np = np.array(image_tensor)

    if input_range == "255":
        image_np = image_np / 255.0

    if input_dim_mode in [1, "chw"]:
        image_np = image_np.transpose(1, 2, 0)
        #: @? (channel, height, width) -> (height, width, channel)

    overlayed_image = image_np.copy()
    height, width, _ = image_np.shape

    for class_id, mask_tensor in segmasks_dict.items():
        if isinstance(mask_tensor, np.ndarray):
            mask_np = mask_tensor
        else:
            #: @PyTorch
            mask_np = mask_tensor.cpu().numpy()

        #: If the mask tensor has 3 dimensions, the first dimension is the batch size:
        if len(mask_np.shape) == 3:
            mask_np = mask_np[0]

        binary_mask = mask_np

        #: Get the color for the class from the class_colors dict or generate a random one and store it
        if class_id not in class_colors:
            class_colors[class_id] = np.random.rand(3)

        color = class_colors[class_id]

        #: Create a 3-channel colored mask
        colored_mask = np.zeros((height, width, 3), dtype=np.float32)
        for ch in range(3):
            # ic(binary_mask.shape, colored_mask[:, :, ch].shape, color[ch])

            my_color = color[ch]
            colored_mask[:, :, ch] = binary_mask * my_color

        #: Alpha blend the colored mask with the overlayed_image
        blended_image = alpha * colored_mask + (1 - alpha) * overlayed_image
        overlayed_image = blended_image * binary_mask[
            ..., np.newaxis
        ] + overlayed_image * (1 - binary_mask[..., np.newaxis])

    overlayed_image_pil = Image.fromarray((overlayed_image * 255).astype(np.uint8))

    return overlayed_image_pil


##
def compute_segmentation_metrics(
    predicted_mask,
    ground_truth_mask,
):
    #: The inputs can be batched or unbatched.
    ##

    # Ensure the masks are binary
    predicted_mask = predicted_mask > 0
    ground_truth_mask = ground_truth_mask > 0

    # Compute True Positive, False Positive, True Negative, False Negative
    TP = (predicted_mask & ground_truth_mask).float().sum().item()
    FP = (predicted_mask & ~ground_truth_mask).float().sum().item()
    TN = (~predicted_mask & ~ground_truth_mask).float().sum().item()
    FN = (~predicted_mask & ground_truth_mask).float().sum().item()

    # Compute metrics
    metrics = {}
    metrics["IoU"] = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0
    metrics["Precision"] = TP / (TP + FP) if TP + FP != 0 else 0
    metrics["Recall"] = TP / (TP + FN) if TP + FN != 0 else 0
    metrics["F1-Score"] = (
        2
        * (metrics["Precision"] * metrics["Recall"])
        / (metrics["Precision"] + metrics["Recall"])
        if metrics["Precision"] + metrics["Recall"] != 0
        else 0
    )
    metrics["Accuracy"] = (TP + TN) / (TP + FP + FN + TN)

    return metrics


##
