import PIL
import re
import torch
import torchvision
from pynight.common_dict import simple_obj
from pynight.common_numpy import image_url2np
from pynight.common_torch import (
    torch_shape_get,
    model_device_get,
)
from pynight.common_debugging import fn_name_current


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
    bias_token_p=None,
):
    #: @todo We should hardcode some models like DINO.
    #:
    #: * @tests
    #: `patch_info_from_name('vit_base_patch16_clip_224.openai_ft_in12k_in1k')`
    ##
    num_prefix_tokens = 1

    model_size = None
    image_resolution = None
    patch_resolution = None
    model_size_pattern = None
    if model_name == "vit_small_patch14_dinov2":
        patch_resolution = 14
        image_resolution = 518
    elif model_name.startswith("RN") or model_name in (
        "blip",
        "ALIGN",
        "NegCLIP",
        "FLAVA",
        "AltCLIP",
        "ALIGN",
    ):
        #: NA for now, just hardcoding some nonsense
        ##
        patch_resolution = 100
        model_size = "NA"
    else:
        ##
        pat1 = re.compile(r"^(?:(?:coca_|xlm-roberta-[^-]+-)?ViT|EVA\d*)-[^-]+-(\d+)")
        pat1_model_size = re.compile(
            r"^(?:(?:coca_|xlm-roberta-[^-]+-)?ViT|EVA\d*)-([^-]+)-\d+"
        )
        ##

        model_size_pattern = None
        if model_name.startswith("vit_"):
            patch_pattern = r"patch(\d+)"
            resolution_pattern = r"_(\d{3,})"
        elif pat1.match(model_name):
            #: =ViT-B-32_laion400m_e31=
            patch_pattern = pat1
            model_size_pattern = pat1_model_size
            resolution_pattern = None
        elif model_name.startswith("mixer_"):
            #: 'mixer_b16_224.goog_in21k_ft_in1k'
            ##
            num_prefix_tokens = 0

            patch_pattern = r"mixer_b(\d+)"
            resolution_pattern = r"mixer_b\d+_(\d{3,})"
        else:
            raise NotImplementedError(
                f"{fn_name_current()}: unsupported model name: {model_name}"
            )

        # Find patch resolution
        patch_match = re.search(patch_pattern, model_name)
        patch_resolution = int(patch_match.group(1)) if patch_match else None
        assert patch_resolution is not None

        if resolution_pattern is not None:
            # Find image resolution
            resolution_match = re.search(resolution_pattern, model_name)
            image_resolution = (
                int(resolution_match.group(1)) if resolution_match else None
            )
            assert image_resolution is not None
        else:
            image_resolution = None

    if image_resolution is not None:
        # Compute the patch count
        patch_count_fl = (image_resolution**2) / (patch_resolution**2)
        patch_count = int(patch_count_fl)
        assert patch_count_fl == patch_count
        patch_count += num_prefix_tokens  #: for CLS
    else:
        patch_count = None

    model_size = None
    if model_size_pattern is not None:
        m = model_size_pattern.search(model_name)
        if m:
            model_size = m.group(1)

    output = dict(
        model_name=model_name,
        model_size=model_size,
        image_resolution=image_resolution,
        patch_resolution=patch_resolution,
        num_prefix_tokens=num_prefix_tokens,
    )

    if bias_token_p is not None and patch_count is not None:
        source_count = patch_count
        if bias_token_p:
            source_count += 1

        output.update(
            patch_count=patch_count,
            source_count=source_count,
            bias_token_p=bias_token_p,
        )

    return simple_obj(**output)


##
def image_from_url(*, model, url, device=None):
    if device is None:
        device = model_device_get(model)
    elif device == "NA":
        device = None

    if isinstance(url, str):
        image_np = image_url2np(url=url)

        image_pil = torchvision.transforms.ToPILImage()(image_np)
    elif isinstance(url, PIL.Image.Image):
        image_pil = url
    else:
        raise ValueError(f"unsupported type for: {url}")

    transforms = model_transform_get(model)
    image_transformed_tensor = transforms.transform_tensor(image_pil).cpu()
    image_transformed = transforms.transform(image_pil)

    image_cpu_squeezed = image_transformed
    image_cpu = image_cpu_squeezed.unsqueeze(0)

    if device:
        image_dv = image_cpu.to(device)
    else:
        image_dv = None

    return simple_obj(
        # image_np=image_np,
        image_pil=image_pil,
        image_natural=image_transformed_tensor,
        image_cpu_squeezed=image_cpu_squeezed,
        image_cpu=image_cpu,
        image_dv=image_dv,
    )


def image_batch_from_urls(*, model, urls, device=None):
    if device is None:
        device = model_device_get(model)
    elif device == "NA":
        device = None

    image_objects = [image_from_url(model=model, url=url, device="NA") for url in urls]

    #: Assuming all images are of same dimensions
    image_batch_cpu = torch.stack(
        [image_obj.image_cpu.squeeze(0) for image_obj in image_objects]
    )

    image_natural = [image_obj.image_natural for image_obj in image_objects]

    if device:
        image_batch_dv = image_batch_cpu.to(device)
    else:
        image_batch_dv = None

    return simple_obj(
        image_natural=image_natural,
        image_batch_cpu=image_batch_cpu,
        image_batch_dv=image_batch_dv,
    )


##
