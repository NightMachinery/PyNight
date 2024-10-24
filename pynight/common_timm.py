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
def model_transform_get(model):
    import timm

    if hasattr(model, "pretrained_cfg"):
        data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
        # ic(data_cfg)

        transform = timm.data.create_transform(**data_cfg)

    elif hasattr(model, "visual"):
        from open_clip.transform import (
            image_transform_v2,
            AugmentationCfg,
            PreprocessCfg,
            merge_preprocess_dict,
            merge_preprocess_kwargs,
        )

        pp_cfg = PreprocessCfg(**model.visual.preprocess_cfg)

        transform = image_transform_v2(
            pp_cfg,
            is_train=False,
        )

    else:
        raise NotImplemenetedError(
            f"{fn_name_current()}: could not find the transforms needed by the model"
        )

    transforms = transform.transforms
    #: =transforms= is Compose and =.transforms= gives us a list of all the transforms in it.

    if isinstance(transforms[-1], torchvision.transforms.Normalize):
        transform_tensor = torchvision.transforms.Compose(transforms[:-1])
        transform_normalize = transforms[-1]

    else:
        print(
            f"{fn_name_current()}: There was no Normalize transform for the current model!",
            flush=True,
        )

        transform_tensor = torchvision.transforms.Compose(transforms)
        transform_normalize = torchvision.transforms.ToTensor
        #: ToTensor is effectively identity

    return simple_obj(
        transform=transform,
        transform_tensor=transform_tensor,
        transform_normalize=transform_normalize,
    )


##
def model_name_get(model, mode="arch+tag"):
    if hasattr(model, "name"):
        return model.name

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
    if (
        model_name == "vit_small_patch14_dinov2"
        or "patch14_dinov2.lvd142m" in model_name
    ):
        patch_resolution = 14
        image_resolution = 518

    elif model_name in [
        "gmixer_24_224.ra3_in1k",
    ]:
        num_prefix_tokens = 0

        patch_resolution = 16
        image_resolution = 224

    elif model_name in [
        "mambaout_small.in1k",
        "mambaout_tiny.in1k",
    ]:
        num_prefix_tokens = 0

        patch_resolution = 16  #: @?
        image_resolution = 224

    elif model_name in [
        "ViT-SO400M-14-SigLIP",
        "ViT-SO400M-14-SigLIP.OC.webli",
    ] or model_name.startswith("vit_so400m_patch14_siglip_224"):
        num_prefix_tokens = 0

        patch_resolution = 14
        image_resolution = 224

    elif model_name in [
        "vit_so400m_patch14_siglip_378.webli_ft_in1k",
        "vit_so400m_patch14_siglip_gap_378.webli_ft_in1k",
    ]:
        num_prefix_tokens = 0

        patch_resolution = 14
        image_resolution = 378

    elif model_name.startswith("ViT-SO400M-14-SigLIP-384"):
        num_prefix_tokens = 0

        patch_resolution = 14
        image_resolution = 384

    elif model_name in [
        "flexivit_large.1200ep_in1k",
    ]:
        patch_resolution = 16
        image_resolution = 240

    elif model_name.startswith("EVA02-E-14"):
        patch_resolution = 14
        image_resolution = 224

    # elif model_name.startswith("RN") or model_name in (
    #     "blip",
    #     "ALIGN",
    #     "NegCLIP",
    #     "FLAVA",
    #     "AltCLIP",
    #     "ALIGN",
    # ):
    #     #: NA for now, just hardcoding some nonsense
    #     ##
    #     patch_resolution = 100
    #     model_size = "NA"

    else:
        ##
        pat1 = re.compile(r"^(?:(?:coca_|xlm-roberta-[^-]+-)?ViT|EVA\d*)-[^-]+-(\d+)")
        pat1_model_size = re.compile(
            r"^(?:(?:coca_|xlm-roberta-[^-]+-)?ViT|EVA\d*)-([^-]+)-\d+"
        )
        ##

        model_size_pattern = None
        if any(
            re.search(pat, model_name)
            for pat in [
                "^vit_",
                "^deit3_",  #: We need to set num_prefix_tokens to 2 for DeiT1
                "^beit(?:v2)?_",
                "^eva(?:02)?_",
            ]
        ):
            #: `eva02_tiny_patch14_336.mim_in22k_ft_in1k`
            ##
            patch_pattern = r"patch(\d+)"
            resolution_pattern = r"_(\d{3,})"

        elif pat1.match(model_name):
            #: =ViT-B-32_laion400m_e31=
            patch_pattern = pat1
            model_size_pattern = pat1_model_size
            # resolution_pattern = None
            resolution_pattern = r"\d{2}-(\d{3,})"
            #: EVA02-L-14-336

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
def image_from_url(*, model, url, accept_gray_p=True, device=None):
    if device is None:
        device = model_device_get(model)

    elif device == "NA":
        device = None

    if isinstance(url, str):
        image_np = image_url2np(
            url=url,
            accept_gray_p=accept_gray_p,
        )

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


def image_batch_from_urls(*, model, urls, accept_gray_p=True, device=None):
    if device is None:
        device = model_device_get(model)
    elif device == "NA":
        device = None

    image_objects = [
        image_from_url(
            model=model,
            url=url,
            device="NA",
            accept_gray_p=accept_gray_p,
        )
        for url in urls
    ]

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
