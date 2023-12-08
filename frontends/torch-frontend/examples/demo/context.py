# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

import base64
import copyreg
import dataclasses
import io
import hashlib
import os
import torch
import pickle
import logging

from shutil import copytree, rmtree
from torch._prims_common import suggest_memory_format
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from typing import Any, Dict, List, Set, Optional, Tuple

log = logging.getLogger(__name__)

class ByteirContext:
    def __init__(self):
        HOME_DIR = os.getenv("HOME")
        if HOME_DIR == None:
            HOME_DIR = "/tmp/"
        self.CACHE_HOME_DIR = os.path.join(HOME_DIR, ".byteir_cache/compile_cache/")

    def __enter__(self):
        os.environ["ByteirCacheDir"] = self.CACHE_HOME_DIR

    def __exit__(self, exception_type, exception_value, traceback):
        del os.environ["ByteirCacheDir"]


def sha256_hash(data: bytes) -> str:
    # [:51] to strip off the "Q====" suffix common to every hash value.
    return base64.b32encode(hashlib.sha256(data).digest())[:51].decode("utf-8").lower()

@dataclasses.dataclass
class TensorMetadata:
    """
    The Tensor metadata relevant when hashing FxGraph cache keys.
    """
    dtype: torch.dtype
    shape: torch.Size
    stride: Tuple[Any, ...]
    device: torch.device
    layout: torch.layout
    memory_format: Optional[torch.memory_format]
    storage_offset: int
    requires_grad: bool
    is_quantized: bool
    is_conj: bool
    is_neg: bool
    is_coalesced: bool
    dense_dim: int
    sparse_dim: int

def extract_tensor_metadata(t: torch.Tensor) -> TensorMetadata:
    """
    Extract the TensorMetadata of a tensor.
    """
    memory_format: Optional[torch.memory_format] = suggest_memory_format(t)
    if not t.is_contiguous(memory_format=memory_format):
        memory_format = None

    return TensorMetadata(
        dtype=t.dtype,
        shape=t.shape,
        stride=t.stride() if t.layout == torch.strided else (),
        device=t.device,
        layout=t.layout,
        memory_format=memory_format,
        storage_offset=t.storage_offset(),
        requires_grad=t.requires_grad,
        is_quantized=t.is_quantized,
        is_conj=t.is_conj(),
        is_neg=t.is_neg(),
        is_coalesced=t.is_coalesced() if t.is_sparse else False,
        dense_dim=t.dense_dim() if t.is_sparse else False,
        sparse_dim=t.sparse_dim() if t.is_sparse else False,
    )

def _ident(x: Any) -> Any:
    return x

def _reduce_fake_tensor(t):
    """
    See FxGraphCachePickler. Custom reducer to pickle FakeTensors.
    """
    metadata = extract_tensor_metadata(t)
    return (_ident, (metadata,))

def _reduce_tensor(t):
    """
    See FxGraphCachePickler. Custom reducer to pickle Tensors.
    """
    # If we see tensors, we know they're contstants stored as attributes on
    # the GraphModule. See tensor lowering; small constants are inlined. If
    # we see a small tensor, therefore, no reference will ultimately remain
    # in the generated code. So we need to include its value in the cache key.
    # Large constannts are effectively treated as inputs and we consider only
    # their metadata.
    metadata = extract_tensor_metadata(t)
    return (_ident, (metadata,))

def _reduce_symint(s):
    """
    See FxGraphCachePickler. Custom reducer to pickle SymInts.
    """
    # For hashing purposes, we only care about the name of the symbol and
    # not the backed value. We evaluate guards stored with a cached graph
    # to ensure a cached entity with SymInt args is safe to reuse.
    return (_ident, (str(s),))


class FxGraphCachePickler(pickle.Pickler):
    """
    Custom pickler to customize the pickling of some objects (Tensors), only for the
    purpose of computing a hash for keying into the FxGraphCache. Tensors contain
    objects that don't pickle and/or vary between runs, and we want to capture the
    data that allow us to compute a stable, but safe hash.
    """

    dispatch_table = copyreg.dispatch_table.copy()
    dispatch_table[torch._subclasses.fake_tensor.FakeTensor] = _reduce_fake_tensor
    dispatch_table[torch.Tensor] = _reduce_tensor
    dispatch_table[torch.SymInt] = _reduce_symint

    @staticmethod
    def dumps(obj) -> bytes:
        """
        Pickle an object using the FxGraphCachePickler.
        """
        with io.BytesIO() as stream:
            pickler = FxGraphCachePickler(stream)
            pickler.dump(obj)
            return stream.getvalue()

    @staticmethod
    def get_hash(obj: Any) -> str:
        """
        Serialize an object using the FxGraphCachePickler and return a hash
        of the pickled object.
        """
        serialized_data = FxGraphCachePickler.dumps(obj)
        return sha256_hash(serialized_data)

@dataclasses.dataclass
class OrderedSetHolder:
    """
    See FxGraphHashDetails. Holds a sorted list to support stable hashing
    of set kwargs.
    """

    items: List[Any]

class FxGraphHashDetails:
    """
    Object to capture all the details for a compiled FX graph relevant to computing
    a safe and stable cache key.
    """

    # Excluded kwargs param that are not stable between runs
    EXCLUDED_KWARGS = ["graph_id"]

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: List[torch.Tensor],
        fx_kwargs: Dict[str, Any],
    ):
        self.gm = gm
        self.example_inputs = example_inputs

        # Order kwargs so hashing is stable to changes in kwarg order.
        self.fx_kwargs = {}
        for k in sorted(fx_kwargs):
            if k not in self.EXCLUDED_KWARGS:
                if type(fx_kwargs[k]) is set:
                    # Special case to handle set params. Python sets can't be
                    # ordered, so sort the elements and store them in a proxy.
                    self.fx_kwargs[k] = OrderedSetHolder(sorted(fx_kwargs[k]))
                else:
                    self.fx_kwargs[k] = fx_kwargs[k]

    def debug_str(self) -> str:
        """
        Get a printable string describing in more detail all the attributes
        comprising this object. Useful for debugging when one graph hashes
        to a different value than another.
        """

        def get_str(obj) -> str:
            if isinstance(obj, torch.Tensor):
                return str(extract_tensor_metadata(obj))
            elif isinstance(obj, bytes):
                return "<bytes>"
            else:
                return str(obj)

        lines = []
        for attr, obj in vars(self).items():
            if isinstance(obj, list):
                for ii in range(len(obj)):
                    h = FxGraphCachePickler.get_hash(obj[ii])
                    lines.append(f"[{h}] {attr}[{ii}]: {get_str(obj[ii])}")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    h = FxGraphCachePickler.get_hash(v)
                    lines.append(f"[{h}] {attr}[{k}]: {get_str(v)}")
            else:
                h = FxGraphCachePickler.get_hash(obj)
                lines.append(f"[{h}] {attr}: {get_str(obj)}")
        return "\n".join(lines)


def compiled_fx_graph_hash(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    fx_kwargs: Dict[str, Any],
) -> str:
    """
    Generate a unique hash of the FX graph for caching.
    """
    details = FxGraphHashDetails(gm, example_inputs, fx_kwargs)
    # The prefix distinguishes among the other kinds of objects we
    # cache in this module.
    key = "f" + FxGraphCachePickler.get_hash(details)
    log.debug("[byteir] FX graph cache hash details for key %s:\n%s", key, details.debug_str())
    return key


class FxGraphCache:
    """
    Supports caching and reusing compiled Fx graphs.

    The overall strategy is as follows:
    - This cache stores entries on disk. When saving an entry, we can't
      serialize callables (that could be C++, Triton, etc.), so we serialize
      their own disk cache location. We then recreate the compiled artifact
      after fetching from disk.
    - For indexing the cache, we gather the fields relevant to identifying an
      FxGraph (the graph module, graph inputs, system settings etc.) into an
      FxGraphCacheDetails object, pickle it, and compute a hash for the key.
      See FxGraphCachePickler.
    - Among the metadata we store, we also include a guards expression that's
      appropriate for validating any symbols for Tensor arguments that have
      symbolic bounds. On cache lookup then, we evaluate those guards in the
      current context to validate that a cached entry can be served.
    - A given graph could have multiple compiled versions, corresponding to
      different sets of guards. Therefore, we store cache entries in the form:
          <temp dir>/<fx graph hash>/<serialized metatdata>
    - On lookup, we compute the key from the graph details, iterate over all
      leaf files in the corresponding subdirectory, deserialize the entry, and
      evaluate its guards expression. If the evaluation succeeds, we have a
      cache hit. If it fails, we compile the graph and store a new entry.
    - Finally, on a cache hit, we need to make sure any guards that would
      have been created during compilation are added to the current context.
    """

    # TODO(masnesral): Investigate whether it's beneficial to store compiled graphs
    # in an in-memory cache after loading from disk.
    @staticmethod
    def _get_tmp_dir() -> str:
        """
        Get the toplevel temporary directory for storing compiled graphs.
        """
        return os.path.join(os.environ["ByteirCacheDir"])

    @staticmethod
    def _get_tmp_dir_for_key(key: str) -> str:
        """
        Return the disk location for a given cache key.
        """
        return os.path.join(FxGraphCache._get_tmp_dir(), key)

    @staticmethod
    def _filter_symints(inputs: List[Any]) -> List[torch.SymInt]:
        """
        Get the SymInt objects from the input list.
        """
        return [s for s in inputs if isinstance(s, torch.SymInt)]

    @staticmethod
    def _get_shape_env() -> ShapeEnv:
        """
        Helper to get the shape env from the tracing context.
        """
        return torch._guards.TracingContext.get().fake_mode.shape_env

    @staticmethod
    def get_hash_key(
        graph: torch.fx.GraphModule,
        example_inputs: List[torch.Tensor],
        fx_kwargs: Dict[str, Any],
    ) -> str:
        """
        Get hash key for a given graph
        """
        return compiled_fx_graph_hash(graph, example_inputs, fx_kwargs)

    @staticmethod
    def save_to_cache(
        compiled_rt_folder: str, key: str
    ):
        """
        Move compiled temp runtime folder to cache
        """
        cache_dir = FxGraphCache._get_tmp_dir_for_key(key)
        copytree(compiled_rt_folder, cache_dir, dirs_exist_ok=True)

    @staticmethod
    def try_load(
        key: str, rt_folder: str
    ):
        """
        Load a compiled graph from the cache, return True on cache hit
        """
        #from filelock import FileLock
        cache_dir = FxGraphCache._get_tmp_dir_for_key(key)
        #lock_dir = get_lock_dir()
        #lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
        #with lock:
        if os.path.exists(cache_dir):
            log.debug("[byteir] fx graph cache hit for key %s", key)
            copytree(cache_dir, rt_folder, dirs_exist_ok=True)
            return True
        else:
            log.debug("[byteir] fx graph cache miss for key %s", key)
            return False

    @staticmethod
    def clear():
        """
        Clear out the on-disk cache.
        """
        rmtree(FxGraphCache._get_tmp_dir())
