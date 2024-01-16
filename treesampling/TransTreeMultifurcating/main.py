import math
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.distributions.log_normal import LogNormal


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create the positional encodings
        position = torch.arange(max_len).unsqueeze(1)
        # div_term = torch.exp(
        #     torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        # )
        # pe = torch.zeros(1, max_len, d_model)
        # pe[0, :, 0::2] = torch.sin(position * div_term)
        # pe[0, :, 1::2] = torch.cos(position * div_term)
        div_term_full = torch.exp(
            torch.arange(0, d_model) * (-math.log(10000.0) / d_model)
        )
        div_term_sin = div_term_full[::2]
        div_term_cos = div_term_full[1::2]
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term_sin)
        pe[0, :, 1::2] = torch.cos(position * div_term_cos)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


n_nodes = 13
torch.manual_seed(2)
device = "cuda:0"
scale = 1  # scaling of the embedding vs number of nodes

enc_layer = nn.TransformerEncoderLayer(
    d_model=scale * n_nodes,
    nhead=n_nodes,
    batch_first=True,
    dim_feedforward=1024,
    dropout=0.1,
)
layer_norm_enc = nn.LayerNorm(scale * n_nodes)
enc = nn.TransformerEncoder(enc_layer, num_layers=2, norm=layer_norm_enc).to(device)

Pos_enc = PositionalEncoding(scale * n_nodes, 0.0, max_len=n_nodes + 1).to(device)

dec_layer = nn.TransformerDecoderLayer(
    d_model=scale * n_nodes,
    nhead=n_nodes,
    batch_first=True,
    dropout=0.1,
    dim_feedforward=1024,
)
layer_norm_dec = nn.LayerNorm(scale * n_nodes)
dec = nn.TransformerDecoder(dec_layer, num_layers=5, norm=layer_norm_dec).to(device)
# dec_from = nn.Linear(n_nodes, n_nodes).to(device)
# dec_to = nn.Linear(n_nodes, n_nodes).to(device)

dec_from = nn.Sequential(
    nn.Linear(scale * n_nodes, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, n_nodes),
).to(device)
dec_to = nn.Sequential(
    nn.Linear(scale * n_nodes, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, n_nodes),
).to(device)

enc_trans = nn.Linear(n_nodes, n_nodes * scale).to(device)
dec_trans = nn.Linear(n_nodes, n_nodes * scale).to(device)

activation = nn.ReLU()
softmax = nn.Softmax()

opt = torch.optim.Adam(
    [
        {
            "params": enc.parameters(),
        },
        {
            "params": dec.parameters(),
        },
        {
            "params": dec_from.parameters(),
        },
        {
            "params": dec_to.parameters(),
        },
        {
            "params": dec_trans.parameters(),
        },
        {
            "params": enc_trans.parameters(),
        },
    ],
    lr=0.000001,
    amsgrad=True,
)


def print_params(model):
    tot = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(name, torch.norm(param.data))
            tot += torch.norm(param.data)

    print("Total norm: ", tot)


print_params(enc)
print_params(dec)


def vimco(log_p, logq_tau, inverse_temp=1.0):
    log_prior = math.log((1 / (n_nodes ** (n_nodes - 2))))

    bs = log_p.shape[0]
    if bs == 1:
        print("K must be greater than 1 to use VIMCO!")
    log_p_joint = log_p + log_prior
    log_ratio = log_p_joint - logq_tau
    # log_ratio = log_p - logq_tau
    mean_exclude_signal = (torch.sum(log_ratio) - log_ratio) / (bs - 1.0)
    control_variates = torch.logsumexp(
        log_ratio.view(-1, 1).repeat(1, bs)
        - log_ratio.diag()
        + mean_exclude_signal.diag()
        - np.log(bs),
        dim=0,
    )
    temp_lower_bound = torch.logsumexp(log_ratio - np.log(bs), dim=0)
    vimco_fake_term = torch.sum(
        (temp_lower_bound - control_variates).detach() * logq_tau, dim=0
    )

    return vimco_fake_term


def score_function_estimator(log_p, log_q):
    log_ratio = (log_p - log_q).detach()
    return torch.mean((log_ratio - 1) * log_q)


def generate_trees(W):
    batch_size = W.shape[0]
    logq_tau = torch.zeros(batch_size, device=device)

    W_trans = enc_trans(W)
    W_pos = Pos_enc(W_trans)
    mem = enc(W_pos)
    A = torch.zeros(batch_size, n_nodes, n_nodes, device=device)
    mask_to = torch.zeros(batch_size, n_nodes, device=device)
    mask_to[:, 0] = -torch.inf  # the first node exists
    mask_from = torch.zeros(batch_size, n_nodes, device=device)
    mask_from[:, 1:] = -torch.inf  # first node exists

    for i in range(n_nodes - 1):
        A_with_token = torch.cat(
            [-torch.ones(batch_size, 1, n_nodes, device=device), A], 1
        )
        A_with_token_trans = dec_trans(A_with_token)
        A_pos = Pos_enc(A_with_token_trans)  # (batch, n_nodes+1, n_nodes)
        out = dec(A_pos, mem)
        # out = enc(A_pos)
        # out = dec(A_pos, W)
        logq_to = out[:, 0, :]
        logq_to = dec_to(logq_to)

        logq_to_normed = (logq_to.T - logq_to.logsumexp(dim=1)).T
        logq_to_normed = logq_to_normed + mask_to

        # print("logq_to_normed: ", logq_to_normed[0])
        # print("mask_to: ", mask_to[0])

        updates_to = torch.distributions.Multinomial(1, logq_to_normed.exp()).sample()
        indices_to = torch.nonzero(updates_to)
        # print("indices_to: ", indices_to[0, 1])

        logq_from = out[
            indices_to[:, 0], indices_to[:, 1] + 1, :
        ]  # (batch, assigned nodes)
        logq_from = dec_from(logq_from)

        logq_from_normed = (logq_from.T - logq_from.logsumexp(dim=1)).T
        logq_from_normed = logq_from_normed + mask_from
        updates_from = torch.distributions.Multinomial(
            1, logq_from_normed.exp()
        ).sample()
        indices_from = torch.nonzero(updates_from)
        # print("mask from: ", mask_from[0])
        # print("indices_from: ", indices_from[0, 1])

        # update adj
        A_new = torch.zeros_like(A, device=device)
        A_new = A_new + A
        A_new[
            indices_from[:, 0],
            indices_from[:, 1],
            indices_to[:, 1],
        ] = 1.0
        A = A_new

        # update masks
        mask_to[indices_to[:, 0], indices_to[:, 1]] = -torch.inf
        mask_from[indices_to[:, 0], indices_to[:, 1]] = 0.0

        # remember the prob
        logq_tau = (
            logq_tau
            + logq_from_normed[updates_from == 1.0]
            + logq_to_normed[updates_to == 1.0]
        )

    return A[:, :n_nodes, :], logq_tau


def generative_model(W, A):
    logp = (W * A).sum(dim=(1, 2))

    return logp


def get_w(batch_size=10, node_size=n_nodes, shifted=False, sparse=1.0, std=1.0):
    W = ((torch.randn(1, n_nodes, n_nodes) * std) ** 2).to(device)

    # manage sparsity
    if sparse < 1.0:
        eps = 0.01
        W *= torch.rand_like(W) < sparse
        W += eps

    if shifted:
        W[:, 0, :] *= 0.1
        W[:, 0, 2] += 100

    W = W / W.sum(dim=(1, 2), keepdim=True)

    # W = W.log()

    # W = W / W.sum(dim=(1, 2), keepdim=True)

    return W.repeat(batch_size, 1, 1)


# batch_size = 512
batch_size = 256
# batch_size = 128
# batch_size = 56
W_tester = get_w(batch_size=batch_size, shifted=True, sparse=0.7)
print("W_tester", W_tester[0], W_tester.shape)

hist_p = np.array([])
hist_q = np.array([])
hist_loss = np.array([])
################## TRANING
init_inverse_temp = 0.001
warm_start_intervals = 10000
for iter in range(200000):
    inverse_temp = min(1.0, init_inverse_temp + iter * 1.0 / warm_start_intervals)
    W_temp = get_w(batch_size=batch_size, sparse=0.9)

    # W_temp = W_tester
    A, logq_tau = generate_trees(W_temp)
    logp = generative_model(W_temp, A)
    loss = -vimco(logp, logq_tau, inverse_temp=inverse_temp)
    # loss = score_function_estimator(logp, logq_tau)

    # loss = -score_function_estimator(logp, logq_tau)
    opt.zero_grad()
    loss.backward()
    opt.step()

    hist_p = np.append(hist_p, logp.mean().item())
    hist_q = np.append(hist_q, logq_tau.mean().item())
    hist_loss = np.append(hist_loss, loss.mean().item())
    if iter % 100 == 0:
        with torch.no_grad():
            A, logq_tau = generate_trees(W_tester)
            logp = generative_model(W_tester, A)
            loss = -vimco(logp, logq_tau)  # without temp

        print(
            f"Iter: {iter} Loss: ",
            loss.mean().item(),
            "Logp: ",
            logp.mean().item(),
            "Logp rolling avg:",
            hist_p[-10:].mean(),
            "Logp max: ",
            logp.max().item(),
        )

################### FINISHED TRAINING
A = A.to("cpu")
print("PROB TAU: ", logq_tau)


########### PLOTTING


plt.plot(hist_p)
plt.savefig("train_logp.png")
plt.close()

plt.plot(hist_q)
plt.savefig("train_logq.png")
plt.close()

plt.plot(hist_loss)
plt.savefig("train_loss.png")
plt.close()

largest_q = logq_tau.argmax().item()
largest_p = logp.argmax().item()
graphs = list(range(min(10, batch_size)))
graphs.append(largest_q)
graphs.append(largest_p)
print(graphs)

for n, i in enumerate(graphs):
    print(f"plotting, enum={n} index={i}, q={logq_tau[i].item()}, p={logp[i].item()}")
    A_tmp = A[i, :, :]
    G = nx.from_numpy_array(A_tmp.numpy(), create_using=nx.DiGraph())

    plt.figure(figsize=(9, 9))
    plt.title(
        f"Graph {n} {i}, \nlogq(tau)= {logq_tau[i].item()}, \nlogp(tau)={logp[i].item()}"
    )
    nx.draw(
        G,
        nx.planar_layout(G),
        with_labels=True,
        # node_color=colors,
        node_size=1000,
        font_size=20,
        font_weight="bold",
    )
    plt.savefig(f"graph_{n}.png")
