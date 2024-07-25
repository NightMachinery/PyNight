import torch
import torch.nn.functional as F
import re

from pynight.common_icecream import ic
from pynight.common_iterable import to_iterable
from pynight.common_dict import simple_obj
from pynight.common_datasets import (
    TransformResult,
)
from pynight.common_torch import (
    rank_tensor,
)


##
def normalize_map(
    attributions,
    normalize,
    # outlier_quantile=0.1,
    outlier_quantile=0,
    num_prefix_tokens=0,
    bias_token_p=False,
    clone_p=True,
    pixel_p=False,
):
    if pixel_p:
        attributions_skipped = attributions
    else:
        attributions_skipped = attributions[..., num_prefix_tokens:]  #: Skips CLS
        if bias_token_p:
            attributions_skipped = attributions_skipped[..., :-1]
            #: Skips the bias token

        # ic(num_prefix_tokens, attributions.shape, attributions_skipped.shape)

    attributions_normalized = attributions_skipped
    if clone_p:
        attributions_normalized = attributions_normalized.clone()

    normalize = to_iterable(normalize)

    if normalize:
        for normalizer in normalize:
            #: Computing these quantiles is expensive, so we do it only when needed.
            if normalizer in [
                "scale_by_max_signed_attr",
            ]:
                #: higher outlier_quantile makes discerning the attributions in the middle easier, but the higher attributions become the same
                if outlier_quantile == 0:
                    max_attr = torch.max(
                        attributions_normalized,
                        dim=-1,
                        keepdim=True,
                    ).values
                else:
                    max_attr = torch.quantile(
                        attributions_normalized,
                        1 - outlier_quantile,
                        dim=-1,
                        keepdim=True,
                    )

                # max_attr = torch.max(attributions_normalized).item()
            else:
                max_attr = None

            if normalizer in [
                "scale_by_max_signed_attr",
                "shift_min_to_zero",
            ]:
                if outlier_quantile == 0:
                    min_attr = torch.min(
                        attributions_normalized,
                        dim=-1,
                        keepdim=True,
                    ).values
                else:
                    min_attr = torch.quantile(
                        attributions_normalized,
                        outlier_quantile,
                        dim=-1,
                        keepdim=True,
                    )

                # min_attr = torch.min(attributions_normalized).item()
            else:
                min_attr = None

            if normalizer in [
                "scale_by_max_abs_attr",
            ]:
                if outlier_quantile == 0:
                    max_abs_attr = torch.max(
                        torch.abs(attributions_normalized),
                        dim=-1,
                        keepdim=True,
                    ).values
                else:
                    max_abs_attr = torch.quantile(
                        torch.abs(attributions_normalized),
                        1 - outlier_quantile,
                        dim=-1,
                        keepdim=True,
                    )

                # max_abs_attr = torch.max(torch.abs(attributions_normalized)).item()
            else:
                max_abs_attr = None

            # ic(
            #     attributions_normalized.shape,
            #     max_attr.shape,
            #     min_attr.shape,
            #     max_abs_attr.shape,
            # )
            # attributions_normalized.shape: torch.Size([10, 196])
            # max_attr.shape: torch.Size([10, 1])
            # min_attr.shape: torch.Size([10, 1])
            # max_abs_attr.shape: torch.Size([10, 1])

            if normalizer == "scale_by_max_abs_attr":
                attributions_normalized /= max_abs_attr

            elif normalizer.lower() == "relu":
                attributions_normalized = F.relu(attributions_normalized)

            elif normalizer == "scale_by_max_signed_attr":
                attributions_normalized = torch.where(
                    attributions_normalized < 0,
                    attributions_normalized / torch.abs(min_attr),
                    attributions_normalized / max_attr,
                )

            elif normalizer == "shift_min_to_zero":
                attributions_normalized -= min_attr
                attributions_normalized = torch.relu(attributions_normalized)
            
            elif normalizer == "rank_uniform":
                ranks_top_is_1 = rank_tensor(
                    attributions_normalized,
                    dim=-1,
                    descending=True,
                    increment=1,
                )
                #: The most important patch is turned into =1=, and the least important patch is turned into =len(attr)=.

                attributions_normalized = 1.0 - (
                    (ranks_top_is_1 - 1.0) / (ranks_top_is_1.shape[-1] - 1.0)
                )
                #: =ranks_top_is_1 - 1= makes the ranks_top_is_1 start from zero.
                #: =ranks_top_is_1.shape[-1] - 1= adjusts the denominator for the above change.
                #: =1 - ...= reverses the order of the ranking, so that the first rank becomes 1, and the last rank becomes 0.

            else:
                raise ValueError(f"Unsupported normalizer: {normalizer}")

    return simple_obj(
        attributions_skipped=attributions_skipped,
        attributions_normalized=attributions_normalized,
    )


##
def transform_attr_threshold(
    batch,
    *,
    attr_name,
    normalize_opts=None,
    gt_threshold=None,
    lt_threshold=None,
    device=None,
    return_mode="mutate",
):
    attr = batch[attr_name]
    attr_new = attr  # .clone()
    #: =normalize_map= clones its input, so we don't need to do so.

    if device is not None:
        attr_new = attr_new.to(device)

    attr_name_new = attr_name

    attr_name_new = re.sub(r"^attributions_s_", "attributions_p_", attr_name_new)
    #: _s stands for "signed scalar", and _p stands for "per-patch".

    # ic(attr_name, attr_name_new)

    if normalize_opts is not None:
        normalize = normalize_opts["normalize"]
        normalize = to_iterable(normalize)
        attr_new = normalize_map(
            attr_new,
            **normalize_opts,
        ).attributions_normalized
        attr_name_new += f"""_N:{'->'.join(normalize)}"""

    if gt_threshold is not None:
        attr_name_new += f"_GT{gt_threshold}"

        mask = attr_new > gt_threshold
        attr_new = mask * attr_new

    if lt_threshold is not None:
        attr_name_new += f"_LT{lt_threshold}"

        mask = attr_name < lt_threshold
        attr_new = mask * attr_new

    if return_mode == "mutate":
        batch[attr_name_new] = attr_new
    elif return_mode == "new_only":
        batch = dict()
        batch[attr_name_new] = attr_new
    else:
        raise ValueError(f"Unsupported return_mode: {return_mode}")

    return TransformResult(
        result=batch,
        name=attr_name_new,
        #: =name= is used in =segment_by_attr=.
    )


##
