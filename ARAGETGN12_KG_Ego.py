# ============================================================
# Agentic Retrieval-Augmented ETGN for Temporal Fraud Detection
# with Knowledge Graph Integration and Ego-Network Subgraphs
#
# Changes relative to ARAGETGN11.py:
#   1. Knowledge Graph (KG) module: raw ERC20 transactions are
#      transformed into a heterogeneous graph G_KG = (V, E, R)
#      containing wallets, tokens, account-type entities, and
#      behavior/attack-pattern entities connected by typed
#      relations. A relation-aware KG encoder (KGNN) produces
#      knowledge embeddings H_K that are fused with the
#      transaction-graph embeddings H_G via H_F = W_f [H_G || H_K]
#      before edge classification.
#   2. The Reachability-subgraph feature pipeline is replaced by
#      an Ego-network subgraph pipeline (radius-2 ego graphs).
#   3. Confusion-matrix plots (raw counts and row-normalized) of
#      true labels against predicted labels are produced during
#      evaluation.
# ============================================================

import pandas as pd
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from transformers import MixtralModel, MixtralConfig

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 200
EGO_RADIUS = 2

# ============================================================
# Knowledge Graph Construction
#
# G_KG = (V, E, R)
#   V : heterogeneous entities
#         - Wallet   (every from/to address)
#         - Token    (every tokenAddress)
#         - AccountType entities (EOA, Contract)
#         - Behavior/attack-pattern entities
#           (HighFanOut, HighFanIn, HighValue)
#   R : relation types
#         0 TRANSFERS      Wallet  -> Wallet
#         1 USES_TOKEN     Wallet  -> Token      (sender side)
#         2 CREDITS        Token   -> Wallet     (receiver side)
#         3 IS_TYPE        Wallet  -> EOA/Contract entity
#         4 INTERACTS_WITH Wallet  -> Contract wallet
#         5 EXHIBITS       Wallet  -> Behavior entity
# ============================================================

REL_TRANSFERS = 0
REL_USES_TOKEN = 1
REL_CREDITS = 2
REL_IS_TYPE = 3
REL_INTERACTS_WITH = 4
REL_EXHIBITS = 5
NUM_RELATIONS = 6

KG_FEAT_DIM = 8  # [is_wallet, is_token, is_type_ent, is_pattern_ent,
                 #  in_deg_norm, out_deg_norm, log_value_norm, is_contract]


def build_knowledge_graph(slice_data, node_order):
    """
    Build the heterogeneous fraud knowledge graph for one time slice.

    slice_data : DataFrame with the raw ERC20 transaction columns
    node_order : list of wallet addresses in the exact order used by
                 the transaction graph (G.nodes), so that kg_map[i]
                 gives the KG index of transaction-graph node i.

    Returns (kg_x, kg_edge_index, kg_edge_type, kg_map) as tensors.
    """
    entity_index = {}

    def get_entity(key):
        if key not in entity_index:
            entity_index[key] = len(entity_index)
        return entity_index[key]

    # ---- Wallet entities (registered first, in transaction-graph order)
    for addr in node_order:
        get_entity(("wallet", addr))

    # ---- Account-type entities
    eoa_ent = get_entity(("type", "EOA"))
    contract_ent = get_entity(("type", "CONTRACT"))

    # ---- Behavior / attack-pattern entities
    fanout_ent = get_entity(("pattern", "HIGH_FANOUT"))
    fanin_ent = get_entity(("pattern", "HIGH_FANIN"))
    highval_ent = get_entity(("pattern", "HIGH_VALUE"))

    # ---- Token entities
    for tok in slice_data["tokenAddress"].unique():
        get_entity(("token", tok))

    edges, rels = [], []

    # Per-wallet aggregates for features and pattern relations
    out_deg = defaultdict(int)
    in_deg = defaultdict(int)
    val_sum = defaultdict(float)
    is_contract_flag = defaultdict(int)

    log_values = np.log1p(slice_data["value"].astype(float).values)
    high_value_cut = np.percentile(log_values, 90) if len(log_values) else 0.0

    senders = slice_data["from"].values
    receivers = slice_data["to"].values
    tokens = slice_data["tokenAddress"].values
    from_is_c = slice_data["fromIsContract"].values
    to_is_c = slice_data["toIsContract"].values

    for s, r, t, fc, tc, logv in zip(senders, receivers, tokens,
                                     from_is_c, to_is_c, log_values):
        u = get_entity(("wallet", s))
        v = get_entity(("wallet", r))
        tok = get_entity(("token", t))

        # Transaction relation
        edges.append((u, v)); rels.append(REL_TRANSFERS)

        # Token relations (sender uses token, token credits receiver)
        edges.append((u, tok)); rels.append(REL_USES_TOKEN)
        edges.append((tok, v)); rels.append(REL_CREDITS)

        # Contract interaction
        if tc == 1:
            edges.append((u, v)); rels.append(REL_INTERACTS_WITH)

        # High-value behavior
        if logv >= high_value_cut:
            edges.append((u, highval_ent)); rels.append(REL_EXHIBITS)

        out_deg[s] += 1
        in_deg[r] += 1
        val_sum[s] += float(logv)
        val_sum[r] += float(logv)
        if fc == 1:
            is_contract_flag[s] = 1
        if tc == 1:
            is_contract_flag[r] = 1

    # Account-type and fan-out/fan-in pattern relations
    out_vals = list(out_deg.values()) or [0]
    in_vals = list(in_deg.values()) or [0]
    fanout_cut = np.percentile(out_vals, 90)
    fanin_cut = np.percentile(in_vals, 90)

    for addr in node_order:
        w = entity_index[("wallet", addr)]
        type_ent = contract_ent if is_contract_flag[addr] == 1 else eoa_ent
        edges.append((w, type_ent)); rels.append(REL_IS_TYPE)
        if out_deg[addr] >= max(fanout_cut, 2):
            edges.append((w, fanout_ent)); rels.append(REL_EXHIBITS)
        if in_deg[addr] >= max(fanin_cut, 2):
            edges.append((w, fanin_ent)); rels.append(REL_EXHIBITS)

    # ---- Node features X for every KG entity
    num_entities = len(entity_index)
    kg_x = np.zeros((num_entities, KG_FEAT_DIM), dtype=np.float32)

    max_out = max(out_vals) or 1
    max_in = max(in_vals) or 1
    max_val = max(val_sum.values()) if val_sum else 1.0

    for (etype, key), idx in entity_index.items():
        if etype == "wallet":
            kg_x[idx, 0] = 1.0
            kg_x[idx, 4] = in_deg[key] / max_in
            kg_x[idx, 5] = out_deg[key] / max_out
            kg_x[idx, 6] = val_sum[key] / max_val if max_val > 0 else 0.0
            kg_x[idx, 7] = float(is_contract_flag[key])
        elif etype == "token":
            kg_x[idx, 1] = 1.0
        elif etype == "type":
            kg_x[idx, 2] = 1.0
        else:  # pattern
            kg_x[idx, 3] = 1.0

    kg_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    kg_edge_type = torch.tensor(rels, dtype=torch.long)
    kg_map = torch.arange(len(node_order), dtype=torch.long)  # wallets registered first

    return torch.tensor(kg_x), kg_edge_index, kg_edge_type, kg_map


# ============================================================
# PyG Data subclass so KG tensors batch correctly:
# kg_edge_index and kg_map must be offset by the number of KG
# entities (kg_x rows), not by the number of transaction nodes.
# ============================================================
class KGData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in ("kg_edge_index", "kg_map"):
            return self.kg_x.size(0)
        return super().__inc__(key, value, *args, **kwargs)


# ============================================================
# Knowledge Graph Encoder (KGNN)
#
# Relation-aware attention encoder: each relation type r has a
# learned embedding used as an edge feature, so attention weights
# depend on [W h_i || W h_j || r_ij] as in a KG attention network.
# Produces H_K = KGNN(A_K, X_K).
# ============================================================
class KnowledgeGraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_relations=NUM_RELATIONS,
                 rel_dim=8, heads=4):
        super().__init__()
        self.rel_emb = nn.Embedding(num_relations, rel_dim)
        self.kgat1 = GATConv(in_dim, hidden_dim, heads=heads,
                             concat=True, edge_dim=rel_dim)
        self.kgat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1,
                             concat=False, edge_dim=rel_dim)

    def forward(self, kg_x, kg_edge_index, kg_edge_type):
        e = self.rel_emb(kg_edge_type)
        h = F.relu(self.kgat1(kg_x, kg_edge_index, edge_attr=e))
        h = self.kgat2(h, kg_edge_index, edge_attr=e)
        return h


# ============================================================
# Agentic Memory
# ============================================================
class AgenticMemory:
    def __init__(self, memory_size=3000):
        self.memory = []
        self.max_size = memory_size

    def add(self, emb):
        if len(self.memory) >= self.max_size:
            self.memory.pop(0)
        self.memory.append(emb.detach().cpu())

    def retrieve(self, query, k=5):
        if len(self.memory) == 0:
            return None

        mem = torch.stack(self.memory).to(query.device)
        query = F.normalize(query, dim=-1)
        mem = F.normalize(mem, dim=-1)
        scores = torch.matmul(mem, query.unsqueeze(-1)).squeeze(-1)
        idx = torch.topk(scores, min(k, len(scores))).indices
        return mem[idx].mean(dim=0)


# ============================================================
# Retrieval Agent
# ============================================================
class RetrievalAgent(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, emb):
        return self.policy(emb).mean()


# ============================================================
# MGGPT Encoder
# ============================================================
class MGGPT(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads

        # GAT Layers
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)

        # Projection layer to align dimensions
        self.proj = nn.Linear(hidden_dim * heads, hidden_dim)

        # GPT-5 style Encoder by MixtralModel
        cfg = MixtralConfig(
            vocab_size=8,
            hidden_size=hidden_dim,
            num_hidden_layers=4,
            num_attention_heads=heads,
            intermediate_size=hidden_dim * 4,
            num_experts=8,
            num_experts_per_tok=2
        )
        self.gpt = MixtralModel(cfg)

        self.combine = nn.Linear(hidden_dim * 2, hidden_dim)

        # Missing information layers
        self.Wg1 = nn.Linear(hidden_dim, hidden_dim)
        self.Wg2 = nn.Linear(hidden_dim, hidden_dim)
        self.Wb1 = nn.Linear(hidden_dim, hidden_dim)
        self.Wb2 = nn.Linear(hidden_dim, hidden_dim)

    def predict_missing(self, h_v, h_N_v):
        gamma = torch.tanh(self.Wg1(h_v) + self.Wg2(h_N_v))
        beta = torch.tanh(self.Wb1(h_v) + self.Wb2(h_N_v))
        return h_v + (gamma + 1) * beta - h_N_v

    def forward(self, x, edge_index, batch):
        # First GAT Layer
        x = F.relu(self.gat1(x, edge_index))

        # Projection to hidden_dim
        x_proj = self.proj(x)

        # Second GAT
        h_N_v = self.gat2(x, edge_index)

        # Missing information correction
        x_corr = self.predict_missing(x_proj, h_N_v)

        # Batch padding for GPT
        num_graphs = batch.max().item() + 1
        max_nodes = torch.bincount(batch).max().item()
        padded = torch.zeros((num_graphs, max_nodes, self.hidden_dim), device=x.device)

        for i in range(num_graphs):
            idx = (batch == i).nonzero(as_tuple=True)[0]
            padded[i, :len(idx)] = x_corr[idx]

        gpt_out = self.gpt(inputs_embeds=padded).last_hidden_state

        flat = torch.cat(
            [gpt_out[i, :torch.sum(batch == i)] for i in range(num_graphs)], dim=0
        )

        fused = self.combine(torch.cat([x_corr, flat], dim=-1))
        return fused


# ============================================================
# Edge Classifier
# ============================================================
class EdgeClassifier:
    def __init__(self):
        self.model = SGDClassifier(
            loss="log_loss",
            max_iter=2000,
            tol=1e-3,
            random_state=100
        )
        self.is_fitted = False

    def fit(self, X, y):
        """Initial training"""
        self.model.partial_fit(X, y, classes=np.array([0, 1]))
        self.is_fitted = True

    def partial_fit(self, X, y):
        """Online update (optional for streaming graphs)"""
        if not self.is_fitted:
            self.fit(X, y)
        else:
            self.model.partial_fit(X, y)

    def predict_proba(self, X):
        """Return probability of positive class"""
        probs = self.model.predict_proba(X)
        return probs[:, 1]


# ============================================================
# Agentic Retrieval-Augmented ETGN with KG Fusion
#
#   H_G = MGGPT(A, X)               transaction-graph embedding
#   H_K = KGNN(A_K, X_K)            knowledge-graph embedding
#   H_F = ReLU(W_f [H_G || H_K])    fusion (per wallet node)
#   edge embedding = [H_F(u) || H_F(v)]
# ============================================================
class EnhancedTemporalGraphNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.encoder = MGGPT(input_dim, hidden_dim).to(DEVICE)
        self.kg_encoder = KnowledgeGraphEncoder(KG_FEAT_DIM, hidden_dim).to(DEVICE)
        self.kg_fusion = nn.Linear(hidden_dim * 2, hidden_dim).to(DEVICE)

        self.memory = AgenticMemory()
        self.agent = RetrievalAgent(hidden_dim * 2).to(DEVICE)
        self.fusion = nn.Linear(hidden_dim * 4, hidden_dim * 2).to(DEVICE)

        self.edge_classifier = EdgeClassifier()
        self.num_layers = num_layers

    def create_edge_embeddings(self, h, edge_index):
        return torch.cat([h[edge_index[0]], h[edge_index[1]]], dim=1)

    def agentic_fusion(self, edge_emb, train=True):
        prob = self.agent(edge_emb)

        if prob > 0.5:
            ctx = self.memory.retrieve(edge_emb)
            if ctx is not None:
                edge_emb = F.relu(self.fusion(torch.cat([edge_emb, ctx], dim=-1)))

        if train:
            self.memory.add(edge_emb)

        return edge_emb

    def forward(self, x, edge_index, batch,
                kg_x, kg_edge_index, kg_edge_type, kg_map, train=True):
        # Transaction-graph embedding H_G
        h_g = self.encoder(x, edge_index, batch)

        # Knowledge-graph embedding H_K, aligned to wallet nodes
        h_k_all = self.kg_encoder(kg_x, kg_edge_index, kg_edge_type)
        h_k = h_k_all[kg_map]

        # Fusion  H_F = ReLU(W_f [H_G || H_K])
        h = F.relu(self.kg_fusion(torch.cat([h_g, h_k], dim=-1)))

        edge_embeddings = self.create_edge_embeddings(h, edge_index)

        enhanced = []
        for emb in edge_embeddings:
            enhanced.append(self.agentic_fusion(emb, train))

        return torch.stack(enhanced), h


# ============================================================
# ---------------- Graph Construction
# ============================================================
def create_graph(data):
    G = nx.DiGraph()
    nodes = set(data['from'].tolist() + data['to'].tolist())
    G.add_nodes_from(nodes)
    for _, row in data.iterrows():
        G.add_edge(row['from'], row['to'], weight=row['timestamp'])
    return G


# ============================================================
# ---------------- Fraud and Antifraud Scores Evaluation
# ============================================================
def calculate_fraud_and_antifraud_scores(G):
    # Centrality-based scores
    base_fraud = nx.eigenvector_centrality(G, max_iter=1000)
    base_antifraud = nx.out_degree_centrality(G)

    # Degree-based scores (normalized)
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    max_in = max(in_deg.values()) if in_deg else 1
    max_out = max(out_deg.values()) if out_deg else 1

    in_deg_norm = {n: v / max_in for n, v in in_deg.items()}
    out_deg_norm = {n: v / max_out for n, v in out_deg.items()}

    # Final combined scores
    fraud_scores = {
        n: base_fraud.get(n, 0.0) + out_deg_norm.get(n, 0.0)
        for n in G.nodes()
    }

    antifraud_scores = {
        n: base_antifraud.get(n, 0.0) + in_deg_norm.get(n, 0.0)
        for n in G.nodes()
    }

    return fraud_scores, antifraud_scores


# ============================================================
# ---------------- Labeling Nodes
# ============================================================
def label_nodes(fraud_scores, antifraud_scores, fraud_threshold=0.01, antifraud_threshold=0.01):
    labels = {}
    for node in fraud_scores:
        collection_label = 'collection_irregular' if fraud_scores[node] > fraud_threshold else 'collection_regular'
        pay_label = 'pay_regular' if antifraud_scores[node] > antifraud_threshold else 'pay_irregular'
        labels[node] = (collection_label, pay_label)
    return labels


# ============================================================
# ---------------- Creating Ego-Network subgraph
#
# Replaces the Reachability-subgraph approach. The ego network of
# a node is the subgraph induced by all nodes within EGO_RADIUS
# hops of the node (both incoming and outgoing directions), plus
# the node itself.
# ============================================================
def create_ego_network(G, node, radius=EGO_RADIUS):
    return nx.ego_graph(G, node, radius=radius, undirected=True)


# ============================================================
# ---------------- Building Ego networks for all nodes
# ============================================================
def build_ego_networks(G, radius=EGO_RADIUS):
    ego_networks = {}
    for node in G.nodes:
        ego_networks[node] = create_ego_network(G, node, radius)
    return ego_networks


# ============================================================
# ---------------- Counting labeled nodes inside an Ego network
# ============================================================
def count_ego_labels(ego_net, labels, label):
    count = 0
    for n in ego_net.nodes:
        if n in labels and label in labels[n]:
            count += 1
    return count


# ============================================================
# ---------------- Feature Extraction from Ego networks
# ============================================================
def extract_features(G, node, ego_networks, labels):
    ego_net = ego_networks[node]

    R1 = count_ego_labels(ego_net, labels, 'collection_regular')
    R2 = count_ego_labels(ego_net, labels, 'collection_irregular')
    R3 = count_ego_labels(ego_net, labels, 'pay_regular')
    R4 = count_ego_labels(ego_net, labels, 'pay_irregular')

    ego_size = ego_net.number_of_nodes()
    ego_density = nx.density(ego_net) if ego_size > 1 else 0.0

    return [
        R1, R2, ego_size,
        R3, R4, ego_density,
        G.in_degree(node), G.out_degree(node)
    ]


# ============================================================
# ---------------- Create Edge Labels
# ============================================================
def create_edge_labels(G, labels, edge_index, node_to_idx):
    y = []
    idx_to_node = {v: k for k, v in node_to_idx.items()}
    for i in range(edge_index.shape[1]):
        u = idx_to_node[edge_index[0, i].item()]
        label = labels[u][0] == 'collection_irregular'
        y.append(int(label))
    return torch.tensor(y, dtype=torch.long)


# ============================================================
# ---------------- Creating Datalist (with KG attributes)
# ============================================================
def create_data_list(G_list, labels_list, slice_list):
    data_list = []

    for G, labels, slice_data in zip(G_list, labels_list, slice_list):
        node_order = list(G.nodes)
        node_to_idx = {node: idx for idx, node in enumerate(node_order)}

        edge_index = torch.tensor(
            [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges],
            dtype=torch.long
        ).t().contiguous()

        # Ego networks are built once per graph, then reused for
        # every node's feature vector.
        ego_networks = build_ego_networks(G, radius=EGO_RADIUS)

        x = torch.tensor(
            [extract_features(G, node, ego_networks, labels) for node in node_order],
            dtype=torch.float
        )

        y = create_edge_labels(G, labels, edge_index, node_to_idx)

        # Knowledge graph for this time slice
        kg_x, kg_edge_index, kg_edge_type, kg_map = build_knowledge_graph(
            slice_data, node_order
        )

        data = KGData(
            x=x, edge_index=edge_index, y=y,
            kg_x=kg_x, kg_edge_index=kg_edge_index,
            kg_edge_type=kg_edge_type, kg_map=kg_map
        )
        data_list.append(data)

    return data_list


# ============================================================
# Training
# ============================================================
def train_model(model, train_loader, epochs=EPOCHS):
    model.train()
    all_emb, all_labels = [], []

    for epoch in range(epochs):
        for data in train_loader:
            data = data.to(DEVICE)
            emb, _ = model(
                data.x, data.edge_index, data.batch,
                data.kg_x, data.kg_edge_index, data.kg_edge_type, data.kg_map,
                train=True
            )
            all_emb.append(emb.detach().cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

        print(f"Epoch {epoch+1}/{epochs} completed.")

    X = np.concatenate(all_emb)
    y = np.concatenate(all_labels)

    model.edge_classifier.fit(X, y)
    print("SGD training completed.")
    return model


# ============================================================
# Confusion Matrix Plots (true labels vs predicted labels)
# ============================================================
def plot_confusion_matrix(labels, binary, out_path="confusion_matrix.png"):
    cm = confusion_matrix(labels, binary)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    class_names = ["Regular", "Irregular"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Greens", cbar=True,
                vmin=0, vmax=1,
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Confusion Matrix (Row-Normalized)")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Confusion matrix saved to {out_path}")
    return cm


# ============================================================
# Evaluation
# ============================================================
def evaluate_model(model, loader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            emb, _ = model(
                batch.x, batch.edge_index, batch.batch,
                batch.kg_x, batch.kg_edge_index, batch.kg_edge_type, batch.kg_map,
                train=False
            )
            prob = model.edge_classifier.predict_proba(emb.cpu().numpy())
            preds.append(prob)
            labels.append(batch.y.cpu().numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    binary = (preds > 0.5).astype(int)

    auc = roc_auc_score(labels, preds)
    precision = precision_score(labels, binary, zero_division=0)
    recall = recall_score(labels, binary, zero_division=0)
    f1 = f1_score(labels, binary, zero_division=0)

    print(f"AUC={auc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    cm = plot_confusion_matrix(labels, binary)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(cm)

    return auc, precision, recall, f1


# ============================================================
# Main
# ============================================================
def main():
    # Accessing the dataset
    file_path = "combined_data_erc20_4_blocks.csv"
    data = pd.read_csv(file_path, nrows=25000)

    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    data = data.sort_values(by='timestamp')

    # time-slicing
    data['time_slice'] = (data['timestamp'] - data['timestamp'].min()).dt.days // 31

    threshold = 20
    time_slice_counts = data['time_slice'].value_counts()
    sparse_time_slices = time_slice_counts[time_slice_counts < threshold].index.tolist()

    if sparse_time_slices:
        combined_time_slice = max(data['time_slice']) + 1
        data.loc[data['time_slice'].isin(sparse_time_slices), 'time_slice'] = combined_time_slice

    data['time_slice'] = data['time_slice'].astype(int)
    new_time_slice_counts = data['time_slice'].value_counts().sort_index()
    print(new_time_slice_counts)

    G_list, labels_list, slice_list = [], [], []

    for ts in data['time_slice'].unique():
        slice_data = data[data['time_slice'] == ts]
        G = create_graph(slice_data)
        fraud_scores, antifraud_scores = calculate_fraud_and_antifraud_scores(G)
        labels = label_nodes(fraud_scores, antifraud_scores)
        G_list.append(G)
        labels_list.append(labels)
        slice_list.append(slice_data)

    (train_G, test_G,
     train_labels, test_labels,
     train_slices, test_slices) = train_test_split(
        G_list, labels_list, slice_list, test_size=0.3, shuffle=False
    )

    train_data = create_data_list(train_G, train_labels, train_slices)
    test_data = create_data_list(test_G, test_labels, test_slices)

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=8)

    model = EnhancedTemporalGraphNetwork(input_dim=8, hidden_dim=16).to(DEVICE)
    model = train_model(model, train_loader)

    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()
