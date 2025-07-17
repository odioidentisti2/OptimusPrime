import torch

def create_edge_adjacency_mask(batched_edge_index):
    num_edges = batched_edge_index.size(1)
    source_nodes = batched_edge_index[0]
    target_nodes = batched_edge_index[1]

    # unsqueeze and expand
    exp_src = source_nodes.unsqueeze(1).expand(-1, num_edges)
    exp_trg = target_nodes.unsqueeze(1).expand(-1, num_edges)

    src_adj = exp_src == exp_src.T
    trg_adj = exp_trg == exp_trg.T
    cross_adj = (exp_src == exp_trg.T) | (exp_trg == exp_src.T)

    adj_mask = src_adj | trg_adj | cross_adj
    # Mask out self-adjacency by setting the diagonal to False
    adj_mask.fill_diagonal_(0)  # We use "0" here to indicate False in PyTorch boolean context

    return adj_mask

def edge_mask(b_ei, b_map, B, L):
    mask = torch.full(size=(B, L, L), fill_value=False, dtype=torch.bool, device=b_ei.device)
    edge_to_graph = b_map.index_select(0, b_ei[0, :])  # graph index for each edge

    edge_adj = create_edge_adjacency_mask(b_ei)

    # Remap edge indices within each graph to consecutive indices 
    # Step 1: Get unique graph indices (sorted), and for each edge its position in the per-graph consecutive list
    unique_graphs, ei_to_original = torch.unique(edge_to_graph, sorted=True, return_inverse=True)
    # ei_to_original is [num_edges], values in 0..(#unique_graphs-1) for each edge in b_ei

    edges = edge_adj.nonzero()  # [num_connected_pairs, 2]
    graph_index = edge_to_graph[edges[:, 0]]
    coord_1 = ei_to_original[edges[:, 0]]
    coord_2 = ei_to_original[edges[:, 1]]

    mask[graph_index, coord_1, coord_2] = True
    mask = mask.unsqueeze(1)
    return ~mask
