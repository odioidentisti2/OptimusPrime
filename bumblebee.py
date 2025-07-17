import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from attention import SAB, PMA
from esa import *
from molecule_dataset import MoleculeDataset




class MAGClassifier(nn.Module):

    def __init__(self, node_dim, edge_dim, hidden_dim=128, num_heads=8, num_inds=32, output_dim=1):
        super(MAGClassifier, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_inds = num_inds
        self.output_dim = output_dim

        # Node and edge encoders
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        # Masked Attention Blocks
        self.edge_attention1 = SAB(hidden_dim * 2, hidden_dim * 2, num_heads, dropout=0.1)
        self.edge_attention2 = SAB(hidden_dim * 2, hidden_dim * 2, num_heads, dropout=0.1)

        # PMA pooling (for each graph in the batch)
        self.pma = PMA(hidden_dim * 2, num_heads, num_seeds=1, dropout=0.1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        # Assume data is a torch_geometric.data.Batch object
        x = data.x                  # [total_nodes, node_dim]
        edge_index = data.edge_index # [2, total_edges]
        edge_attr = data.edge_attr   # [total_edges, edge_dim]
        batch = data.batch           # [total_nodes], maps node idx to graph idx

        # Compute edge_batch: maps each edge to its graph in the batch
        edge_batch = batch[edge_index[0]]   # [total_edges]

        node_feat = self.node_encoder(x)                # [total_nodes, hidden_dim]
        edge_feat = self.edge_encoder(edge_attr)        # [total_edges, hidden_dim]

        src, dst = edge_index
        edge_nodes = torch.cat([node_feat[src], node_feat[dst]], dim=1)  # [total_edges, 2*hidden_dim]
        edge_repr = edge_nodes + torch.cat([edge_feat, edge_feat], dim=1)

        B = batch.max().item() + 1
        L = edge_repr.size(0)  # total number of edges

        # === Build batch-wide edge mask ===
        # b_map: [total_nodes], maps node idx to graph idx
        # edge_index: [2, total_edges]
        # edge_batch: [total_edges], maps edge idx to graph idx

        # edge_mask will compute a [B, L, L] mask where only edges within same graph can attend
        mask = edge_mask(edge_index, batch, B, L)  # [B, L, L]
        mask = mask.unsqueeze(1)  # [B, 1, L, L] for multi-head attention

        # SAB expects [batch, seq, feat], so expand edge_repr for batch dimension
        # We'll need to pad edge_repr to (B, L, 2*hidden_dim) with zeros for missing edges per-graph
        # But if all graphs have same number of edges or if you flatten, you can use [1, L, 2*hidden_dim]
        edge_repr = edge_repr.unsqueeze(0)  # [1, L, 2*hidden_dim]
        edge_repr = self.edge_attention1(edge_repr, adj_mask=mask)
        edge_repr = self.edge_attention2(edge_repr, adj_mask=mask)
        edge_repr = edge_repr.squeeze(0) # [L, 2*hidden_dim]

        # PMA pooling per graph
        # To pool per-graph, we need to group edge representations by edge_batch (graph id)
        graph_embeds = []
        for i in range(B):
            edge_repr_g = edge_repr[edge_batch == i].unsqueeze(0)
            graph_emb = self.pma(edge_repr_g, adj_mask=None)   # [1, 1, 2*hidden_dim]
            graph_embeds.append(graph_emb.squeeze(0).squeeze(0))
        graph_repr = torch.stack(graph_embeds, dim=0)    # [B, 2*hidden_dim]

        logits = self.classifier(graph_repr)    # [B, output_dim]
        return logits.view(-1)                  # [B]

    def forward_old(self, data):
        # data: batch from DataLoader (torch_geometric.data.Batch)
        x = data.x                  # [total_nodes, node_dim]
        edge_index = data.edge_index # [2, total_edges]
        edge_attr = data.edge_attr   # [total_edges, edge_dim]
        batch = data.batch           # [total_nodes]
        edge_batch = self._edge_batch(edge_index, batch) # [total_edges]

        # Encode node and edge features
        node_feat = self.node_encoder(x)                # [total_nodes, hidden_dim]
        edge_feat = self.edge_encoder(edge_attr)        # [total_edges, hidden_dim]

        # Build edge representation: [src_node, dst_node] features + edge features
        src, dst = edge_index                           # each [total_edges]
        edge_nodes = torch.cat([node_feat[src], node_feat[dst]], dim=1)  # [total_edges, 2*hidden_dim]
        edge_repr = edge_nodes + torch.cat([edge_feat, edge_feat], dim=1)

        # For batching, group edges by graph in the batch
        # We'll need to build a mask for each graph's edge set
        out = []
        num_graphs = batch.max().item() + 1
        for graph_idx in range(num_graphs):
            edge_mask = (edge_batch == graph_idx)
            edge_repr_g = edge_repr[edge_mask]              # [num_edges_g, 2*hidden_dim]
            src_g = src[edge_mask]
            dst_g = dst[edge_mask]

            # Edge-edge attention mask for this graph
            # mask[i, j] = True if edge i and edge j share a node
            num_edges = edge_repr_g.size(0)
            if num_edges == 0:
                # Safeguard for graphs with no edges
                graph_emb = torch.zeros(self.hidden_dim * 2, device=edge_repr.device)
                out.append(graph_emb)
                continue
            src_dst = torch.stack([src_g, dst_g], dim=1) # [num_edges, 2]
            mask = torch.zeros((num_edges, num_edges), dtype=torch.bool, device=edge_repr.device)

            # GPU PARALLEL VERSION
            # src_dst: [num_edges, 2]
            src_nodes = src_dst[:, 0:1]  # [num_edges, 1]
            dst_nodes = src_dst[:, 1:2]  # [num_edges, 1]
            # Check for node sharing (broadcasting)
            shared_src = (src_nodes == src_nodes.T) | (src_nodes == dst_nodes.T)
            shared_dst = (dst_nodes == src_nodes.T) | (dst_nodes == dst_nodes.T)
            mask = shared_src | shared_dst  # [num_edges, num_edges]

            # # CPU version
            # for i in range(num_edges):
            #     for j in range(num_edges):
            #         if len(set(src_dst[i].tolist()) & set(src_dst[j].tolist())) > 0:
            #         # If two edges share a node, mask is True (allow attention)
            #             mask[i, j] = True

            mask = mask.unsqueeze(0).unsqueeze(1) # [1, 1, num_edges, num_edges]


            # SAB expects [batch, seq, feat]
            edge_repr_g = edge_repr_g.unsqueeze(0)
            edge_repr_g = self.edge_attention1(edge_repr_g, adj_mask=mask)
            edge_repr_g = self.edge_attention2(edge_repr_g, adj_mask=mask)
            edge_repr_g = edge_repr_g.squeeze(0) # [num_edges, 2*hidden_dim]

            # PMA pooling to get single graph embedding
            edge_repr_g = edge_repr_g.unsqueeze(0)
            graph_emb = self.pma(edge_repr_g, adj_mask=None)   # [1, 1, 2*hidden_dim]
            graph_emb = graph_emb.squeeze(0).squeeze(0)        # [2*hidden_dim]
            out.append(graph_emb)

        # Stack all graph representations into a batch
        graph_repr = torch.stack(out, dim=0)    # [batch_size, 2*hidden_dim]
        logits = self.classifier(graph_repr)    # [batch_size, output_dim]
        return logits.view(-1)                  # [batch_size]

    @staticmethod
    def _edge_batch(edge_index, node_batch):
        # Given edge_index [2, num_edges] and node_batch [num_nodes], 
        # return edge_batch [num_edges] where each edge takes the batch idx of its src node
        # (assumes all edges within a graph)
        return node_batch[edge_index[0]]


def train(model, loader, optimizer, criterion, epoch):
    model.train()  # set training mode
    total_loss = 0
    correct = 0
    total = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        targets = batch.y.view(-1).to(DEVICE)
        optimizer.zero_grad()  # zero gradients
        logits = model(batch)  # forward pass
        loss = criterion(logits, targets)  # calculate loss
        loss.backward()  # backward pass
        optimizer.step()  # update weights

        # calculate accuracy
        total_loss += loss.item() * batch.num_graphs
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == targets).sum().item()
        total += batch.num_graphs
    return total_loss / total, correct / total

def main(): 
    dataset = MoleculeDataset('DATASETS/MUTA_SARPY_4204.csv')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = MAGClassifier(dataset.node_dim, dataset.edge_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(1, NUM_EPOCHS + 1):
        loss, acc = train(model, loader, optimizer, criterion, epoch)
        print(f"Epoch {epoch}: Loss {loss:.4f} Acc {acc:.4f}")
    print("Training complete.")

if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # DEVICE = torch.device('cpu')
    print(f"\nDEVICE: {DEVICE}")
    BATCH_SIZE = 64
    LR = 1e-4
    NUM_EPOCHS = 20 
    # Set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    main()