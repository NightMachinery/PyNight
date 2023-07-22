import evaluate


##
class ConfiguredMetric:
    def __init__(self, metric, *metric_args, **metric_kwargs):
        self.metric = metric
        self.metric_args = metric_args
        self.metric_kwargs = metric_kwargs

    def add(self, *args, **kwargs):
        return self.metric.add(*args, **kwargs)

    def add_batch(self, *args, **kwargs):
        return self.metric.add_batch(*args, **kwargs)

    def compute(self, *args, **kwargs):
        res = self.metric.compute(
            *args, *self.metric_args, **kwargs, **self.metric_kwargs
        )

        res2 = dict()
        for name, v1 in res.items():
            for k, v2 in self.metric_kwargs.items():
                if k not in ["zero_division"]:
                    name += f"_{k}_{v2}"

            # ic(name)
            res2[name] = v1

        return res2

    @property
    def name(self):
        name = self.metric.name

        for k, v in self.metric_kwargs.items():
            name += f"_{k}_{v}"

        return name

    def _feature_names(self):
        return self.metric._feature_names()


##
