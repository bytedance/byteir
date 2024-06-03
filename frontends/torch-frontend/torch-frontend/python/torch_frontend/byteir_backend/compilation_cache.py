################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

import os
import io
import copyreg
import re
import hashlib
import json
import pickle
import getpass
import tempfile
import functools
import logging
import shutil
from copy import copy

from typing import Optional, Any, Callable, Dict, List, Sequence, Tuple, Union
from filelock import FileLock

import torch
from torch._inductor.codecache import (
    LOCK_TIMEOUT,
    write_atomic,
)
from torch._dynamo.utils import counters
from torch.fx.experimental.symbolic_shapes import ShapeEnv, has_hint, hint_int, SYMPY_INTERP
from torch._subclasses.fake_tensor import FakeTensor
from torch.cuda.memory import caching_allocator_alloc, caching_allocator_delete

try:
    import brt
except ImportError:
    ...

from .compiled_function import (CompiledArtifact, ByteIRFunction)
from .utils import (
    dump_tensors_meta_info,
    BypassFxGraphCache,
    OrderedSetHolder,
    TensorMetadata,
    extract_tensor_metadata,
    maybe_get_fake_mode,
    _reduce_fake_tensor,
    _reduce_symint,
    sha256_hash,
)

log = logging.getLogger(__name__)



def get_system_info() -> Dict[str, Any]:
    try:
        system: Dict[str, Any] = {
            "device": {
                "name":
                torch.cuda.get_device_properties(
                    torch.cuda.current_device()).name,
            },
            "version": {
                "cuda": torch.version.cuda,
            },
        }
    except (AssertionError, RuntimeError):
        # If cuda is not installed, none of the above config is relevant.
        system = {}

    system["hash"] = hashlib.sha256(
        json.dumps(system, sort_keys=True).encode("utf-8")).hexdigest()

    return system


class ByteIRFxGraphHashDetails:
    """
    Object to capture all the details for a compiled FX graph relevant to computing
    a safe and stable cache key.
    """

    # Excluded kwargs param that are not stable between runs
    EXCLUDED_KWARGS = ["graph_id", "workdir"]

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

        # # 'Deterministic algorithms' can affect codegen via lowering to cuda kernels.
        # self.deterministic_algorithms_settings = (
        #     torch.are_deterministic_algorithms_enabled(),
        #     torch.is_deterministic_algorithms_warn_only_enabled(),
        #     byteir_backend.deterministic.fill_uninitialized_memory,  # type: ignore[attr-defined]
        # )

        # Global settings affecting matmul codegen.
        self.cuda_matmul_settings = (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        )

        # Also hash on various system info (including the triton compiler version).
        self.torch_version = torch.__version__
        self.system_info = get_system_info()

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
                    h = ByteIRFxGraphCachePickler.get_hash(obj[ii])
                    lines.append(f"[{h}] {attr}[{ii}]: {get_str(obj[ii])}")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    h = ByteIRFxGraphCachePickler.get_hash(v)
                    lines.append(f"[{h}] {attr}[{k}]: {get_str(v)}")
            else:
                h = ByteIRFxGraphCachePickler.get_hash(obj)
                lines.append(f"[{h}] {attr}: {get_str(obj)}")
        return "\n".join(lines)


class ByteIRFxGraphCachePickler(pickle.Pickler):
    """
    Custom pickler to customize the pickling of some objects (Tensors), only for the
    purpose of computing a hash for keying into the FxGraphCache. Tensors contain
    objects that don't pickle and/or vary between runs, and we want to capture the
    data that allow us to compute a stable, but safe hash.
    """

    dispatch_table = copyreg.dispatch_table.copy()
    dispatch_table[FakeTensor] = _reduce_fake_tensor
    # NOTE: upstream _reduce_tensor has bugs in Scalar (like tensor(0)), we directly use
    # _reduce_fake_tensor
    dispatch_table[torch.Tensor] = _reduce_fake_tensor
    dispatch_table[torch.SymInt] = _reduce_symint

    @staticmethod
    def dumps(obj) -> bytes:
        """
        Pickle an object using the ByteIRFxGraphCachePickler.
        """
        with io.BytesIO() as stream:
            pickler = ByteIRFxGraphCachePickler(stream)
            pickler.dump(obj)
            return stream.getvalue()

    @staticmethod
    def get_hash(obj: Any) -> str:
        """
        Serialize an object using the ByteIRFxGraphCachePickler and return a hash
        of the pickled object.
        """
        serialized_data = ByteIRFxGraphCachePickler.dumps(obj)
        return sha256_hash(serialized_data)


def compiled_fx_graph_hash(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    fx_kwargs: Dict[str, Any],
) -> str:
    """
    Generate a unique hash of the FX graph for caching.
    """
    details = ByteIRFxGraphHashDetails(gm, example_inputs, fx_kwargs)
    # The prefix distinguishes among the other kinds of objects we
    # cache in this module.
    key = "f" + ByteIRFxGraphCachePickler.get_hash(details)
    log.debug("FX graph cache hash details for key %s:\n%s", key,
              details.debug_str())
    return key


class ByteIRFxGraphCache:
    """
    Supports caching and reusing compiled MLIR artifact.

    The overall strategy is as follows:
    - This cache stores entries on disk.
    - For indexing the cache, we gather the fields relevant to identifying an
      FxGraph (the graph module, graph inputs, system settings etc.) into an
      FxGraphCacheDetails object, pickle it, and compute a hash for the key.
      See ByteIRFxGraphCachePickler.
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

    base_cache_dir: str = None
    func_table: Dict[str, ByteIRFunction] = {}

    @staticmethod
    def _lookup_func(key: str):
        ByteIRFxGraphCache.func_table[
            key] if key in ByteIRFxGraphCache.func_table else None

    @staticmethod
    def _save_func(key: str, func: ByteIRFunction):
        if key not in ByteIRFxGraphCache.func_table:
            ByteIRFxGraphCache.func_table[key] = func

    @staticmethod
    def _get_tmp_dir() -> str:
        """
        Get the toplevel temporary directory for storing compiled graphs.
        """
        if ByteIRFxGraphCache.base_cache_dir is None:
            sanitized_username = re.sub(r'[\\/:*?"<>|]', "_",
                                        getpass.getuser())
            ByteIRFxGraphCache.base_cache_dir = os.path.join(
                tempfile.gettempdir(),
                "byteir_" + sanitized_username,
            )
            os.makedirs(ByteIRFxGraphCache.base_cache_dir, exist_ok=True)
        #return os.path.join(ByteIRFxGraphCache.base_cache_dir, "byre")
        return ByteIRFxGraphCache.base_cache_dir

    @staticmethod
    def _get_tmp_dir_for_key(key: str) -> str:
        """
        Return the disk location for a given cache key.
        """
        return os.path.join(ByteIRFxGraphCache._get_tmp_dir(), key[1:3], key)

    @staticmethod
    def get_lock_dir() -> str:
        lock_dir = os.path.join(ByteIRFxGraphCache.base_cache_dir, "locks")
        if not os.path.exists(lock_dir):
            os.makedirs(lock_dir, exist_ok=True)
        return lock_dir

    @staticmethod
    def _get_cache_file_name() -> str:
        return "compilation_cache.data"

    @staticmethod
    def _filter_symints(inputs: List[Any]) -> List[torch.SymInt]:
        """
        Get the SymInt objects from the input list.
        """
        return [s for s in inputs if isinstance(s, torch.SymInt)]

    @staticmethod
    def _get_shape_env() -> Optional[ShapeEnv]:
        """
        Helper to get the shape env from the tracing context.
        """
        ctx = torch._guards.TracingContext.get()
        if not ctx:
            return None
        return ctx.fake_mode.shape_env

    @staticmethod
    def _produce_guards_expression(shape_env, placeholders, ignore_static=True):
        """
        Expected to be used with evaluate_guards_expression(). Produces the guards
        for the given placeholders and returns a string expression to be evaluated
        by evaluate_guards_expression given concrete values for the placeholders.
        """
        from torch._dynamo.source import LocalSource
        arg_names = [f"t{i}" for i in range(len(placeholders))]
        guards = shape_env.produce_guards(placeholders, [LocalSource(a) for a in arg_names], ignore_static=ignore_static)
        if guards:
            return " and ".join(guards)
        return None

    @staticmethod
    def _evaluate_guards_expression(code, args):
        """
        Expected to be used with produce_guards_expression(). Evaluates an expression
        generated by produce_guards_expression for the given concrete args.
        """
        arg_names = [f"t{i}" for i in range(len(args))]
        return eval(code, SYMPY_INTERP, {"L": dict(zip(arg_names, args))})

    @staticmethod
    def _lookup_compiled_artifact(
        key: str,
        example_inputs: List[torch.Tensor],
    ) -> Optional[CompiledArtifact]:
        """
        Lookup a compiled graph in the cache by key. On a hit, return the
        deserialized CompiledArtifact object. On a miss, return None.
        """
        subdir = ByteIRFxGraphCache._get_tmp_dir_for_key(key)
        if not os.path.exists(subdir):
            return None

        shape_env = ByteIRFxGraphCache._get_shape_env()
        assert shape_env is not None

        symints = ByteIRFxGraphCache._filter_symints(example_inputs)
        assert all(has_hint(s) for s in symints)
        hints = [hint_int(s) for s in symints]

        # Iterate over any entries in the subdir for this key and evaluate
        # their guards to determine whether there's a hit.
        artifact = None

        cache_file_path = os.path.join(
            subdir, ByteIRFxGraphCache._get_cache_file_name())
        for _path in [cache_file_path]:
            if not os.path.exists(_path):
                continue
            with open(_path, "rb") as f:
                candidate: CompiledArtifact = pickle.load(f)

            if not candidate.guards_expr:
                # No guards to evaluate, so this is a hit.
                artifact = candidate
                break

            # Evaluate the guard expression in the current context.
            # If there's not a cache hit, we don't want the evaluation to
            # affect the current env, e.g., cause the creation of new guards,
            # so we evaluate with the hints instead of the symbols.
            hit = bool(
                ByteIRFxGraphCache._evaluate_guards_expression(
                    candidate.guards_expr, hints))
            log.debug(
                "fx graph cache key %s evaluating guards [%s] with values %s => hit=%s",
                key,
                candidate.guards_expr,
                hints,
                hit,
            )
            if hit:
                artifact = candidate
                break

        if artifact is None:
            return None

        # Now re-evaluate with the symints to add any guards to the current env.
        if artifact.guards_expr:
            check = bool(
                ByteIRFxGraphCache._evaluate_guards_expression(
                    artifact.guards_expr, symints))
            assert check is True
            log.debug("fx graph cache key %s post-load guards: %s", key,
                      shape_env.guards)

        return artifact

    @staticmethod
    def _save_compiled_artifact(key: str, compiled_artifact: CompiledArtifact,
                                example_inputs: List[torch.Tensor]):
        """
        Store a serialized CompiledArtifact on disk.
        """

        # Before serializing, compute the guard expression that will be used to
        # ensure that a CompiledArtifact is valid when loaded from the cache. It's
        # sufficient to consider only the SymInt args to the fx graph since the
        # Tensor shapes are already captured in the hash for the cache key. Any
        # Tensor arg with a symbolic shape will have a SymInt arg for the graph.
        shape_env = ByteIRFxGraphCache._get_shape_env()
        assert shape_env is not None
        symints = ByteIRFxGraphCache._filter_symints(example_inputs)
        compiled_artifact.guards_expr = ByteIRFxGraphCache._produce_guards_expression(
                shape_env, symints)

        try:
            # FIXME compiled_artifact is not serializable.
            content = pickle.dumps(compiled_artifact)
        except Exception as e:
            log.debug("fx graph cache unable to serialize compiled graph: %s",
                      e)
            counters["byteir"]["fxgraph_cache_pickle_error"] += 1
            return

        subdir = ByteIRFxGraphCache._get_tmp_dir_for_key(key)
        if not os.path.exists(subdir):
            os.makedirs(subdir, exist_ok=True)

        # Use a hash of the serialized CompiledArtifact to get a unique file
        # name. The specific name doesn't matter since a lookup involves
        # iterating over all entries in the parent subdir.
        path = os.path.join(subdir, ByteIRFxGraphCache._get_cache_file_name())
        write_atomic(path, content)

    @staticmethod
    def _check_can_cache():
        """
        Check some conditions that would preclude caching and raise BypassFxGraphCache
        to bypass in case caching is not possible.
        """

        if ByteIRFxGraphCache._get_shape_env() is None:
            # The treatment of guards in the caching implementation requires that
            # we have a shape env.
            log.debug("fx graph cache no shape env")
            raise BypassFxGraphCache()

    @staticmethod
    def Load(compile_fn: Callable, gm: torch.fx.GraphModule,
             example_inputs: List[torch.Tensor], **kwargs) -> ByteIRFunction:
        compiled_func = None

        # gen hash key, generate workdir
        key = compiled_fx_graph_hash(gm, example_inputs, kwargs)
        workdir = ByteIRFxGraphCache._get_tmp_dir_for_key(key)

        lock_path = os.path.join(ByteIRFxGraphCache.get_lock_dir(),
                                 key + ".lock")
        with FileLock(lock_path, timeout=LOCK_TIMEOUT):
            # lookup func table
            compiled_func = ByteIRFxGraphCache._lookup_func(key)
            if compiled_func is not None:
                # `ByteIRFunction` cache hit, early return.
                counters["byteir"]["fxgraph_func_cache_hit"] += 1
                return compiled_func

            # try to lookup `compiled_artifact` from local workdir.
            compiled_artifact = ByteIRFxGraphCache._lookup_compiled_artifact(
                key, example_inputs)

            if compiled_artifact is None:
                counters["byteir"]["fxgraph_artifact_cache_miss"] += 1
                # cache miss, generate compiled function through `compile_fn`.
                compiled_artifact = compile_fn(gm,
                                               example_inputs,
                                               workdir=workdir)
                compiled_artifact.hash_key = key
                ByteIRFxGraphCache._save_compiled_artifact(
                    key, compiled_artifact, example_inputs)
            else:
                counters["byteir"]["fxgraph_artifact_cache_hit"] += 1

            # recover/generate `ByteIRFunction` from artifact obj.
            byre_session = brt.Session(alloc_func=caching_allocator_alloc,
                                       free_func=caching_allocator_delete)
            byre_session.load(compiled_artifact.byre_file)
            compiled_func = ByteIRFunction(byre_session,
                                           compiled_artifact.none_indices)
            # save `ByteIRFunction` obj.
            ByteIRFxGraphCache._save_func(key, compiled_func)

        return compiled_func

    @staticmethod
    def Clear():
        """
        Clear out the on-disk cache.
        """
        try:
            shutil.rmtree(ByteIRFxGraphCache._get_tmp_dir())
        except FileNotFoundError:
            pass
