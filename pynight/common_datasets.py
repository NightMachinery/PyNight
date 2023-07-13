##
def dataset_cache_filenames(dataset, sort_p=True, **kwargs):
    res = list(set(d['filename'] for d in dataset.cache_files))

    if sort_p:
        res.sort(**kwargs)

    return res
##
