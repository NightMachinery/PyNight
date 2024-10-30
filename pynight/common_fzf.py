from iterfzf import iterfzf
from pynight.common_dict import simple_obj
from pynight.common_rtl import (
    contains_persian,
    rtl_reshaper_v1,
    rtl_reshaper_fribidi,
)


def rtl_iterfzf(
    iterable,
    *,
    reshaper_func=None,
    multi=True,
    **iterfzf_kwargs,
):
    iterfzf_kwargs["multi"] = multi

    if reshaper_func is None:
        reshaper_func = rtl_reshaper_v1

    original_items = list(iterable)
    indices = list(range(len(original_items)))

    display_items = []
    for idx, item in zip(indices, original_items):
        if contains_persian(item):
            display_item = reshaper_func(item)
        else:
            display_item = item
        # Prepend index to the item, separated by a tab
        display_items.append(f"{idx}\t{display_item}")

    # Ensure the fzf options include '--with-nth=2..' and '--nth=2..'
    extra_opts = iterfzf_kwargs.get("__extra__", [])
    extra_opts = list(extra_opts)  # Convert to list if it's not already

    # Add '--with-nth=2..' and '--nth=2..' if not already present
    opts_str = " ".join(extra_opts)

    if "--with-nth" not in opts_str:
        extra_opts.append("--with-nth=2..")

    else:
        raise ValueError(
            "The option '--with-nth' must not be specified more than once."
        )

    if "--nth" not in opts_str:
        extra_opts.append("--nth=2..")

    else:
        raise ValueError("The option '--nth' must not be specified more than once.")

    iterfzf_kwargs["__extra__"] = extra_opts

    # Call iterfzf with display_items
    selected_display = iterfzf(display_items, **iterfzf_kwargs)

    if selected_display is None:
        return None

    def parse_selection(sel):
        idx_str, _ = sel.split("\t", 1)
        idx = int(idx_str)
        return idx

    if iterfzf_kwargs.get("multi", False):
        selected_indices = []
        selected_items = []
        if isinstance(selected_display, list):
            for sel in selected_display:
                idx = parse_selection(sel)
                selected_indices.append(idx)
                selected_items.append(original_items[idx])
        else:
            idx = parse_selection(selected_display)
            selected_indices.append(idx)
            selected_items.append(original_items[idx])
        return simple_obj(selected=selected_items, indices=selected_indices)
    else:
        idx = parse_selection(selected_display)
        selected_item = original_items[idx]
        return simple_obj(selected=selected_item, indices=[idx])
