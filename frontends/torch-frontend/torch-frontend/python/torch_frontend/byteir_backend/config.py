import os

byteir_not_use_cache = os.environ.get("BYTEIR_NOT_USE_CACHE") == "1"

# TODO. default not save fx graph.
byteir_save_fxgraph = os.environ.get("BYTEIR_SAVE_FXGRAPH", "1") == "1"

# Converts torch rng ops to their functional philox rng equivalents. Note that
# we functionalize only CUDA rng ops today.
functionalize_rng_ops = False

# can be useful for debugging if we are incorrectly creating meta fake tensors
fake_tensor_allow_meta = os.environ.get("FAKE_ALLOW_META", True)

# Enables optional asserts in hotpath code to check for errors.  If
# you are seeing weird accuracy problems, try turning this on.
# This is currently off by default as it will harm tracing time,
# but it is on by default for aot_eager.
debug_assert = False

debug_partitioner = os.environ.get("AOT_PARTITIONER_DEBUG", False)

static_weight_shapes = True

# Applies CSE to the graph before partitioning
cse = True

# Restricts the amount of computation AOTAutograd can do.
max_dist_from_bw = 3
