import torch
import torch.fx as fx

from .fx_utils import get_aten_target
from .fx_match_utils import get_node_consumer, match_chain

byteir_fusible_patterns = {}
aten = torch.ops.aten


def register_byteir_pattern(name):

    def register(pattern):
        if name in byteir_fusible_patterns.keys():
            raise ValueError("Pattern " + name +
                             " has already been registerd.")
        byteir_fusible_patterns[name] = pattern
        return pattern

    return register


class ByteIRFusiblePattern:

    @classmethod
    def match(cls, node, required_fw_nodes) -> bool:
        raise NotImplementedError

    @classmethod
    def get_pattern_recompute_nodes(cls, node, required_fw_nodes):
        raise NotImplementedError


@register_byteir_pattern("transpose_dot")
class TransposeDotPattern(ByteIRFusiblePattern):

    @classmethod
    def match(cls, node, required_fw_nodes) -> bool:
        post_fusible_ops = [aten.mm, aten.bmm]
        if get_aten_target(node) in [aten.t, aten.transpose]:
            can_fuse = all(
                get_aten_target(user) in post_fusible_ops
                for user in node.users)
            all_fw_node = all(user in required_fw_nodes for user in node.users)
            return (not all_fw_node) and can_fuse
        return False

    @classmethod
    def get_pattern_recompute_nodes(cls, node, required_fw_nodes):
        if cls.match(node, required_fw_nodes):
            return [node]
        return []


@register_byteir_pattern("transpose_reshape_transpose_dot")
class TransposeReshapeTransposeDotPattern(ByteIRFusiblePattern):

    @classmethod
    def match(cls, node, required_fw_nodes) -> bool:
        post_fusible_ops = [aten.mm, aten.bmm, aten.transpose]
        if get_aten_target(node) not in [aten.transpose]:
            return False
        if match_chain(node,
                       target_chain=(aten.transpose, aten.expand, aten.clone,
                                     aten._unsafe_view)):
            expand_node = get_node_consumer(node, 0)
            clone_node = get_node_consumer(expand_node, 0)
            view_node = get_node_consumer(clone_node, 0)
            all_fw_node = all(user in required_fw_nodes
                              for user in view_node.users)
            can_fuse = all(
                get_aten_target(user) in post_fusible_ops
                for user in view_node.users)
            return (not all_fw_node) and can_fuse
        return False

    @classmethod
    def get_pattern_recompute_nodes(cls, node, required_fw_nodes):
        if cls.match(node, required_fw_nodes):
            expand_node = get_node_consumer(node, 0)
            clone_node = get_node_consumer(expand_node, 0)
            view_node = get_node_consumer(clone_node, 0)
            recompute_nodes = [node, expand_node, clone_node, view_node]
            for user in view_node.users:
                if user not in required_fw_nodes:
                    recompute_nodes.append(user)
            return recompute_nodes
        return []


@register_byteir_pattern("transpose_transpose")
class TransposeTransposePattern(ByteIRFusiblePattern):

    @classmethod
    def match(cls, node, required_fw_nodes) -> bool:
        if get_aten_target(node) in [aten.t, aten.transpose]:
            for user in node.users:
                if get_aten_target(user) in [aten.t, aten.transpose]:
                    all_fw_node = all(n in required_fw_nodes
                                      for n in user.users)
                    if not all_fw_node:
                        return True
        return False

    @classmethod
    def get_pattern_recompute_nodes(cls, node, required_fw_nodes):
        if cls.match(node, required_fw_nodes):
            recompute_nodes = [node]
            for user in node.users:
                if get_aten_target(user) == aten.t:
                    recompute_nodes.append(user)
            return recompute_nodes
        return []


@register_byteir_pattern("full_bitwise_not_expand")
class FullBitwiseNotExpandPattern(ByteIRFusiblePattern):

    @classmethod
    def match(cls, node, required_fw_nodes) -> bool:
        if match_chain(node,
                       target_chain=(aten.full, aten.bitwise_not,
                                     aten.expand)):
            return True
        return False

    @classmethod
    def get_pattern_recompute_nodes(cls, node, required_fw_nodes):
        if cls.match(node, required_fw_nodes):
            bitwise_node = get_node_consumer(node, 0)
            expand_node = get_node_consumer(bitwise_node, 0)
            recompute_nodes = [node, bitwise_node, expand_node]
            return recompute_nodes
        return []


# Note: This pattern is temporary.
# It is only used to fix issue that full op(dtype is bool) is not supported in byteir.
@register_byteir_pattern("copy_bitwise_not_expand")
class CopyBitwiseNotExpandPattern(ByteIRFusiblePattern):

    @classmethod
    def match(cls, node, required_fw_nodes) -> bool:
        if match_chain(node,
                       target_chain=(aten._to_copy, aten.bitwise_not,
                                     aten.expand, aten.bitwise_or)):
            bitwise_not_node = get_node_consumer(node, 0)
            expand_node = get_node_consumer(bitwise_not_node, 0)
            bitwise_or_node = get_node_consumer(expand_node, 0)
            return True
        return False

    @classmethod
    def get_pattern_recompute_nodes(cls, node, required_fw_nodes):
        if cls.match(node, required_fw_nodes):
            bitwise_not = get_node_consumer(node, 0)
            expand = get_node_consumer(bitwise_not, 0)
            bitwise_or = get_node_consumer(expand, 0)
            recompute_nodes = [node, bitwise_not, expand, bitwise_or]
            return recompute_nodes
        return []


def greedy_transpose_fusion(joint_graph, required_fw_nodes):
    recompute_nodes = []
    post_fuse_ops = [aten.bmm, aten.mm]
    transparent_ops = [aten.clone, aten._to_copy, aten.expand]
    view_ops = [aten.view, aten._unsafe_view]
    transpose_ops = [aten.t, aten.transpose]
    fusible_tag = {}

    INIT_TAG = 0
    POST_FUSION_TAG = 1
    TRANSPOSE_TAG = 2

    for node in reversed(joint_graph.nodes):
        fusible_tag[node] = INIT_TAG

    for node in reversed(joint_graph.nodes):
        if get_aten_target(
                node) in post_fuse_ops and node not in required_fw_nodes:
            fusible_tag[node] = POST_FUSION_TAG

        if get_aten_target(node) in transparent_ops:
            for user in node.users:
                if user in fusible_tag.keys(
                ) and fusible_tag[user] >= POST_FUSION_TAG:
                    fusible_tag[node] = POST_FUSION_TAG
                    recompute_nodes.append(node)

        if get_aten_target(node) in transpose_ops:
            for user in node.users:
                if user in fusible_tag.keys(
                ) and fusible_tag[user] >= POST_FUSION_TAG:
                    recompute_nodes.append(node)
                    fusible_tag[node] = INIT_TAG

    return recompute_nodes


def get_byteir_recompute_nodes(joint_graph, required_fw_nodes):
    recompute_nodes = []
    recompute_nodes.extend(
        greedy_transpose_fusion(joint_graph, required_fw_nodes))
    for name, pattern in byteir_fusible_patterns.items():
        for node in joint_graph.nodes:
            if node.op == 'output':
                continue
            recompute_nodes.extend(
                pattern.get_pattern_recompute_nodes(node, required_fw_nodes))
    recompute_nodes = list(set(recompute_nodes))
    return recompute_nodes
