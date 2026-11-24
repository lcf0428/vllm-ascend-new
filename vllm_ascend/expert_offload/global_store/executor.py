"""Run NPU copies outside the stream-ordered Host callback thread."""

from concurrent.futures import ThreadPoolExecutor


class TransferExecutor:
    def __init__(self, device, timeout):
        import torch

        self.timeout = timeout
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="expert-materialize",
                                       initializer=torch.npu.set_device, initargs=(device,))

    def run(self, function, *args):
        # Timeout is fatal to the worker. It does not imply DMA cancellation;
        # the store is deliberately kept alive until process teardown.
        return self.pool.submit(function, *args).result(timeout=self.timeout)

    def close(self):
        self.pool.shutdown(wait=True)
