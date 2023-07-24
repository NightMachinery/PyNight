from pynight.common_dict import simple_obj
import re


##
def model_name_get(model, mode="arch+tag"):
    cfg = model.pretrained_cfg

    if mode == "arch+tag":
        return f"{cfg['architecture']}.{cfg['tag']}"
    if mode == "hub_id":
        return cfg["hf_hub_id"]
    else:
        raise ValueError(f"mode not recognized: {mode}")


##
def patch_info_from_name(
    model_name,
    bias_token_p=False,
):
    #: @todo We should hardcode some models like DINO.
    #:
    #: * @tests
    #: `patch_info_from_name('vit_base_patch16_clip_224.openai_ft_in12k_in1k')`
    ##
    patch_pattern = r"patch(\d+)"
    resolution_pattern = r"_(\d{3,})"

    # Find patch resolution
    patch_match = re.search(patch_pattern, model_name)
    patch_resolution = int(patch_match.group(1)) if patch_match else None
    assert patch_resolution is not None

    # Find image resolution
    resolution_match = re.search(resolution_pattern, model_name)
    image_resolution = int(resolution_match.group(1)) if resolution_match else None
    assert image_resolution is not None

    # Compute the patch count
    patch_count_fl = (image_resolution**2) / (patch_resolution**2)
    patch_count = int(patch_count_fl)
    assert patch_count_fl == patch_count
    patch_count += 1  #: for CLS

    source_count = patch_count
    if bias_token_p:
        source_count += 1

    return simple_obj(
        image_resolution=image_resolution,
        patch_resolution=patch_resolution,
        patch_count=patch_count,
        source_count=source_count,
    )


##
