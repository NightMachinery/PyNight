open_clip_sep = ".OC."
# open_clip_sep = "+OC+"

##
def model_name_eva2_p(model_name):
    return model_name is not None and 'eva02' in model_name.lower()

def model_name_clip_p(model_name):
    return model_name is not None and open_clip_sep in model_name

##
