import datetime
import sys
from typing import Any, Callable, Dict, Tuple, TypeVar, cast
from functools import wraps

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import TEST_SKIPS, MultiProcessTestCase, skip_if_lt_x_gpu, TestSkip

# add new skipped test exit code
TEST_SKIPS["torch-version-2.2"] = TestSkip(90, "Need torch version bigger than 2.2")

TestFunc = Callable[[object], object]
T = TypeVar("T")
DEVICE_TYPE = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu"
PG_BACKEND = "nccl" if DEVICE_TYPE == "cuda" else "gloo"

NUM_DEVICES = 4

# We use this as a proxy for "multiple GPUs exist"
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    # when we actually have multiple GPUs, relax the requirement to smaller counts.
    NUM_DEVICES = min(NUM_DEVICES, torch.cuda.device_count())


class DistributedTestBase(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return NUM_DEVICES

    @property
    def backend(self) -> str:
        return PG_BACKEND

    def init_pg(self) -> None:
        if "nccl" in self.backend and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        if self.backend not in ["nccl", "gloo", "mpi", "cpu:gloo,cuda:nccl", "meta"]:
            raise RuntimeError(f"Backend {self.backend} not supported!")

        dist.init_process_group(
            backend=self.backend,
            world_size=self.world_size,
            rank=self.rank,  # pyre-ignore[16]
            init_method=f"file://{self.file_name}",  # pyre-ignore[16]
            timeout=datetime.timedelta(seconds=1200),
        )

        # set device for nccl pg for collectives
        if "nccl" in self.backend:
            torch.cuda.set_device(self.rank)

    def destroy_pg(self) -> None:
        # Wait for all ranks to reach here before starting shutdown.
        # FIXME dist.barrier deadlocks with multiple threads and NCCL: https://github.com/pytorch/pytorch/issues/95895
        # dist.all_reduce(torch.zeros((1,), device="cuda" if torch.cuda.is_available() else "cpu"))
        # FIXME can't use the above all_reduce as it causes hangs on bionic and focal. It hangs:
        #  test_dtensor.py  -- DTensorMeshTest.test_dtensor_device_mesh_device_conversion
        dist.barrier()
        dist.destroy_process_group()

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()


# wrapper to initialize comms (processgroup)
def with_comms(func: TestFunc) -> TestFunc:
    assert func is not None

    @wraps(func)  # pyre-ignore[6]
    def wrapper(self, *args: Tuple[object], **kwargs: Dict[str, Any]) -> None:  # type: ignore[misc]
        # if backend not specified, and cuda available, then use nccl, else gloo
        if torch.cuda.is_available() and torch.cuda.device_count() >= self.world_size:
            self.device_type = "cuda"
        else:
            self.device_type = "cpu"

        self.init_pg()
        func(self, *args, **kwargs)  # type: ignore[misc]
        self.destroy_pg()

    return wrapper


def skip_unless_torch_gpu(method: T) -> T:
    """
    Test decorator which skips the test unless there's a GPU available to torch.

    >>> # xdoctest: +SKIP
    >>> @skip_unless_torch_gpu
    >>> def test_some_method(self) -> None:
    >>>   ...
    """
    # The builtin @skip_if_no_gpu relies on os.environ['WORLD_SIZE'] being set.
    return cast(T, skip_if_lt_x_gpu(NUM_DEVICES)(method))


def skip_unless_torch_version_bigger_than(torch_version: str):
    """
    Test decorator which skips the test unless current torch version is
    bigger than the given number.

    >>> # xdoctest: +SKIP
    >>> @skip_unless_torch_version_bigger_than(torch_version="2.2")
    >>> def test_some_method(self) -> None:
    >>>   ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_torch_version = torch.__version__
            if current_torch_version >= torch_version:
                return func(*args, **kwargs)
            sys.exit(TEST_SKIPS[f"torch-version-{torch_version}"].exit_code)

        return wrapper

    return decorator