import torch, torch.nn as nn, os
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.datasets import Amazon, Planetoid
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, degree, add_self_loops
from spikingjelly.clock_driven import functional, neuron
from spikingjelly.activation_based.monitor import OutputMonitor
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def normalized_adj(graph: Data, add_self_loop: bool = True) -> torch.Tensor:
    if add_self_loop:
        edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.x.shape[0])
        return to_dense_adj(edge_index, max_num_nodes=graph.x.shape[0])[0].to_sparse()
    else:
        return to_dense_adj(graph.edge_index, max_num_nodes=graph.x.shape[0])[
            0
        ].to_sparse()


def normalized_degree(
    graph: Data, exponet: float = -0.5, add_self_loop: bool = False
) -> torch.Tensor:
    if add_self_loop:
        edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.x.shape[0])
    else:
        edge_index = graph.edge_index
    degrees = degree(edge_index[0], num_nodes=graph.x.shape[0])
    degrees = torch.pow(degrees, exponet)
    degrees[torch.isinf(degrees)] = 0
    return torch.diag(degrees).to_sparse()


def gcn_conv(graph: Data) -> torch.Tensor:
    A = normalized_adj(graph)
    D = normalized_degree(graph)
    S = torch.sparse.mm((torch.sparse.mm(D, A)), D)
    H = torch.sparse.mm(S, torch.sparse.mm(S, graph.x))
    return H.to_dense()


def positional_encoding(
    data: Data, dim: int = 10, method: str = "random_walk"
) -> torch.Tensor:
    assert method in [
        "random_walk",
        "laplace",
    ], 'Only support "random_walk" or "laplace".'
    A = normalized_adj(data, add_self_loop=False)
    if method == "random_walk":
        D = normalized_degree(data, exponet=-1.0, add_self_loop=False)
        P = torch.sparse.mm(A, D)
        PE = torch.zeros(data.x.shape[0], dim)
        PE[:, 0] = 1.0
        for i in range(1, dim):
            PE[:, i] = torch.diag(torch.sparse.mm(P, PE[:, i - 1].unsqueeze(1)))
    elif method == "laplace":
        D = normalized_degree(data, exponet=-0.5, add_self_loop=False)
        laplacian = torch.eye(data.x.shape[0]) - torch.sparse.mm(
            (torch.sparse.mm(D, A)), D
        )
        _, eigen_vectors = torch.linalg.eigh(laplacian)
        PE = eigen_vectors[:, 1 : dim + 1]
        sign = torch.randint(0, 2, (dim,)) * 2 - 1
        PE = PE * sign.unsqueeze(0)
    return PE


def load_data(
    name: str,
    rpath: str = "datasets",
    split: List[float] = [0.7, 0.15, 0.15],
    batch_size: int = 32,
    positional: bool = False,
    positional_method: str = "random_walk",
    positional_dim: int = 16,
    edge_keep_ratio: float = 0.1,
) -> Tuple[int, int, DataLoader, DataLoader, DataLoader]:
    assert name in [
        "cora",
        "citeseer",
        "pubmed",
        "amazon_photo",
        "amazon_computers",
        "ogbn-arxiv",
    ], 'Only support "cora", "citeseer", "pubmed", "amazon_photo", "amazon_computers" or "ogbn-arxiv"'
    if name == "cora" or name == "citeseer" or name == "pubmed":
        data = Planetoid(rpath, name, split="full")[0]
    elif name == "amazon_photo":
        data = Amazon(rpath, "photo")[0]
    elif name == "amazon_computers":
        data = Amazon(rpath, "computers")[0]
    elif name == "ogbn-arxiv":
        data = PygNodePropPredDataset("ogbn-arxiv", rpath)[0]

    num_edges = data.edge_index.shape[1]
    num_keep = int(num_edges * edge_keep_ratio)
    perm = torch.randperm(num_edges)
    selected_indices = perm[:num_keep]
    data.edge_index = data.edge_index[:, selected_indices]

    H = gcn_conv(data)
    y = data.y.reshape(data.y.shape[0])

    if positional:
        pe = positional_encoding(data, positional_dim, positional_method)
        H = torch.concat((H, pe), dim=1)
    num_features, num_classes = H.shape[-1], int(data.y.max() + 1)

    train_inputs, test_inputs, train_labels, test_labels = train_test_split(
        H, y, test_size=split[1] + split[2], random_state=25
    )
    eval_inputs, test_inputs, eval_labels, test_labels = train_test_split(
        test_inputs,
        test_labels,
        test_size=split[2] / (split[1] + split[2]),
        random_state=25,
    )

    train_set = TensorDataset(train_inputs, train_labels)
    eval_set = TensorDataset(eval_inputs, eval_labels)
    test_set = TensorDataset(test_inputs, test_labels)

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size, shuffle=True)

    return num_features, num_classes, train_loader, eval_loader, test_loader


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    train_loader: DataLoader,
    reset: bool = False,
    device="cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.long().to(device)

        optimizer.zero_grad()
        spike_freq = model(imgs)

        loss = criterion(spike_freq, labels)
        loss.backward()
        optimizer.step()
        if reset is True:
            functional.reset_net(model)

        total_loss += loss.item() * imgs.size(0)
        predicted = spike_freq.argmax(dim=-1)
        correct += (predicted == labels).sum().item()
        total_samples += imgs.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy


def eval_epoch(
    model: nn.Module,
    criterion: nn.CrossEntropyLoss,
    eval_loader: DataLoader,
    reset: bool = False,
    device="cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, labels in eval_loader:
            imgs = imgs.to(device)
            labels = labels.long().to(device)

            spike_freq = model(imgs)
            loss = criterion(spike_freq, labels)
            if reset is True:
                functional.reset_net(model)

            total_loss += loss.item() * imgs.size(0)
            predicted = spike_freq.argmax(dim=-1)
            correct += (predicted == labels).sum().item()
            total_samples += imgs.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy


def test_epoch(
    model: nn.Module,
    criterion: nn.CrossEntropyLoss,
    test_loader: DataLoader,
    reset: bool = False,
    noise: Tuple[float, float] = (0, 0),
    dropout: float = 0.0,
    device="cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    dropout_layer = nn.Dropout(dropout)

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.long().to(device)

            random_noise = torch.normal(
                noise[0], noise[1], imgs.shape, device=imgs.device
            )
            imgs = dropout_layer(imgs)
            imgs += random_noise

            spike_freq = model(imgs)
            loss = criterion(spike_freq, labels)
            if reset is True:
                functional.reset_net(model)

            total_loss += loss.item() * imgs.size(0)
            predicted = spike_freq.argmax(dim=-1)
            correct += (predicted == labels).sum().item()
            total_samples += imgs.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    epochs: int,
    reset: bool = False,
    save_path: str = "outputs",
    file_name: str = "best_model.pth",
    device: str = "cpu",
) -> Tuple[List[float], List[float], List[float], List[float]]:
    train_losses = []
    train_accs = []
    eval_losses = []
    eval_accs = []
    best_acc = 0.0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, optimizer, criterion, train_loader, reset, device
        )
        eval_loss, eval_acc = eval_epoch(model, criterion, eval_loader, reset, device)

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Train accuracy: {(train_acc * 100):.2f}% | Eval Loss: {eval_loss:.4f} | Eval accuracy: {(eval_acc * 100):.2f}%"
        )

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        eval_losses.append(eval_loss)
        eval_accs.append(eval_acc)

        if eval_acc > best_acc:
            best_acc = eval_acc
            model_path = os.path.join(save_path, file_name)
            torch.save(model.state_dict(), model_path)
            print(
                f"Best model saved at epoch {epoch + 1} with Eval Accuracy: {(eval_acc * 100):.2f}%"
            )

    return train_losses, eval_losses, train_accs, eval_accs


def test_model(
    model: nn.Module,
    criterion: nn.CrossEntropyLoss,
    test_loader: DataLoader,
    reset: bool = False,
    dropout: float = 0.0,
    device="cpu",
) -> Dict[Tuple[float, float], float]:
    accs = {}
    noise_settings = [
        (0, 0),
        (0, 0.05),
        (0, 0.1),
        (0, 0.2),
        (0, 0.3),
        (0, 0.4),
        (0, 0.5),
    ]

    for noise in noise_settings:
        if noise == (0, 0):
            sub_accs = []
            for _ in range(20):
                _, acc = test_epoch(
                    model, criterion, test_loader, reset, noise, dropout, device
                )
                sub_accs.append(acc)
            acc = torch.mean(torch.tensor(sub_accs))
            std = torch.std(torch.tensor(sub_accs))
        else:
            _, acc = test_epoch(
                model, criterion, test_loader, reset, noise, dropout, device
            )
        accs[noise] = acc

    return accs, std


def draw_curve(
    train_losses: list,
    eval_losses: list,
    train_accs: list,
    eval_accs: list,
    save_path: str = "outputs",
    file_name: str = "curve.png",
    dpi: int = 100,
) -> None:
    assert len(train_losses) == len(
        eval_losses
    ), "Length mismatch between train and eval losses"
    assert len(train_accs) == len(
        eval_accs
    ), "Length mismatch between train and eval accuracies"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(train_losses, label="Train Loss", color="blue")
    ax1.plot(eval_losses, label="Eval Loss", color="orange", linestyle="--")
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_accs, label="Train Accuracy", color="blue")
    ax2.plot(eval_accs, label="Eval Accuracy", color="orange", linestyle="--")
    ax2.set_title("Accuracy Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    plt.savefig(os.path.join(save_path, file_name), dpi=dpi)
    plt.close()


def consumption(
    model: nn.Module,
    test_loader: DataLoader,
    reset: bool = False,
    EAC: float = 0.9e-12,
    device: str = "cpu",
):
    model.eval()
    if reset:
        monitor = OutputMonitor(model, neuron.LIFNode)
        total_spikes = 0
        total_samples = 0
    else:
        model.net.spike_count_total = torch.zeros(sum(model.net.if_set_layer_potential))
        model.net.tensor_count_total = torch.zeros(
            sum(model.net.if_set_layer_potential)
        )
    firing_rate = 0.0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            model(imgs)
            if reset is True:
                total_spikes += sum(record.sum().item() for record in monitor.records)

                T = len(monitor.records)
                N = monitor.records[0].shape[0]
                C = monitor.records[0].shape[1]
                total_samples += T * N * C

                monitor.clear_recorded_data()
                functional.reset_net(model)
    firing_rate = (
        total_spikes / total_samples
        if reset
        else (model.net.spike_count_total / model.net.tensor_count_total).item()
    )
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()

    return EAC * model.flops * model.time_steps * firing_rate, num_params
