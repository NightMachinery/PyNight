open_clip_sep = ".OC."
# open_clip_sep = "+OC+"


##
def model_name_eva2_p(model_name):
    return model_name is not None and "eva02" in model_name.lower()


def model_name_clip_p(model_name):
    return model_name is not None and open_clip_sep in model_name

def model_name_mixer_p(model_name):
    return model_name is not None and "mixer" in model_name


def model_needs_MLP_DU_p(model_name: str) -> bool:
    #: @todo Add MambaOut

    specific_models = {
        "gmixer_24_224.ra3_in1k",
        "vit_giant_patch14_dinov2",
    }
    return model_name_eva2_p(model_name) or model_name in specific_models


##
