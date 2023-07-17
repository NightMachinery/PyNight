import datasets
import evaluate
import pynight.common_tqdm as common_tqdm


def hf_tqdm_set(tqdm_lib=None, do_enable=True):
    if tqdm_lib is None:
        tqdm_lib = common_tqdm

    datasets.utils.logging.tqdm_lib = tqdm_lib
    evaluate.utils.logging.tqdm_lib = tqdm_lib

    if do_enable:
        datasets.enable_progress_bar()
        evaluate.enable_progress_bar()
