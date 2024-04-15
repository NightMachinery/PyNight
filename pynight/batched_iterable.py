from collections.abc import Mapping
from pynight.common_icecream import ic
from pynight.common_dict import (
    SimpleObject,
    simple_obj,
    BatchedDict,
)


class BatchedIterable:
    def __init__(
        self,
        data,
        batch_size,
        drop_last_batch=False,
        skip_none_p=True,
        autoadjust_batch_size_mode=False,
        # autoadjust_batch_size_mode=True,
    ):
        self.data = data
        self.batch_size = batch_size
        self.drop_last_batch = drop_last_batch
        self.skip_none_p = skip_none_p
        self.autoadjust_batch_size_mode = autoadjust_batch_size_mode
        self.extras = []
        #: We will drop the default empty `extras`, which might not be summable with `res`. Its only purpose is to make using `len(self.extras)` easy without having to check for None.

    def __iter__(self):
        length = len(self.data)
        i = 0

        while i < length or len(self.extras) >= self.batch_size:
            # ic(type(self.extras))

            if len(self.extras) >= self.batch_size:
                #: Yield a batch from extras
                res = self.extras[: self.batch_size]
                self.extras = self.extras[self.batch_size :]
                yield res
                continue

            advance_by = self.batch_size - len(self.extras)
            res = self.data[i : i + advance_by]
            # ic(type(res))
            # ic(len(res), advance_by, len(self.extras))

            i += advance_by

            if self.skip_none_p and res is None:
                continue

            if self.autoadjust_batch_size_mode:
                res_size = len(res)
                if res_size < self.batch_size and self.autoadjust_batch_size_mode in [
                    True,
                    "grow",
                ]:
                    #: Merge with extras
                    if len(self.extras) >= 1:
                        res = self.extras + res

                    if len(res) >= self.batch_size:
                        self.extras = res[self.batch_size :]
                        res = res[: self.batch_size]
                    else:
                        self.extras = res
                        continue
                elif res_size > self.batch_size and self.autoadjust_batch_size_mode in [
                    True,
                    "shrink",
                ]:
                    #: Keep extras for the next batch
                    if len(self.extras) >= 1:
                        self.extras += res[self.batch_size :]
                    else:
                        self.extras = res[self.batch_size :]

                    res = res[: self.batch_size]

                #: This assertion should be true, you can enable it for debugging:
                # if self.autoadjust_batch_size_mode is True:
                #     assert (
                #         len(res) == self.batch_size
                #     ), f"autoadjust_batch_size_mode has not adjusted the size: len={len(res)}, should_be={self.batch_size}"

            # ic(len(res), self.batch_size, self.autoadjust_batch_size_mode)
            yield res

        if not self.drop_last_batch and len(self.extras) > 0:
            #: Yield the remaining extras as the last batch
            yield self.extras

    def __len__(self):
        length = len(self.data)
        num_batches = length // self.batch_size
        if not self.drop_last_batch:
            num_batches += length % self.batch_size != 0
        return num_batches


##
