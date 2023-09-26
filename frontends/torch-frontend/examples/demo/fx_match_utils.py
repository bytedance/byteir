import torch
from compile_utils import get_aten_target

aten = torch.ops.aten

def is_used_by_specific_consumer(node, consumer_type=None):
    if consumer_type == None:
        return True

    all_users = list(node.users)
    if len(all_users) != 1:
        return False
    consumer = all_users[0]
    if not isinstance(consumer_type, (list, tuple)):
        consumer_type = [consumer_type]
    if get_aten_target(consumer) not in consumer_type:
        return False
    return True


def get_node_consumer(node, index):
    all_users = list(node.users)
    return all_users[index]


def match_chain(node, target_chain):
    if len(target_chain) == 1:
        return get_aten_target(node) in target_chain

    if len(list(node.users)) != 1:
        return False

    specific_types = target_chain[0]
    
    if not isinstance(specific_types, (list, tuple)):
        specific_types = [specific_types]

    if get_aten_target(node) in specific_types:
        return match_chain(get_node_consumer(node, 0), target_chain[1:])
    return False
