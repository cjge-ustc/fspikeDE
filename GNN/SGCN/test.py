import argparse, os, torch, io
import torch.nn as nn
from typing import Tuple
from pprint import pformat
from torch.utils.data import DataLoader
from models import SpikeDEGCN, SpikingJellyGCN
from utils import load_data, test_model, consumption


def task_setup(
    args: argparse.Namespace,
) -> Tuple[int, int, DataLoader, DataLoader, DataLoader]:
    num_features, num_classes, train_loader, eval_loader, test_loader = load_data(
        args.dataset,
        "datasets",
        args.split,
        args.batch_size,
        True if args.task == "DRSGCN" else False,
        args.positional_method,
        args.positional_dim,
        edge_keep_ratio=args.edge_keep_ratio,
    )
    print(
        f"{'='*50}\nTask Setup:\nDataset: {args.dataset}\nSplit Ratio: {args.split}\nType: {args.task}\nNumber of Classes: {num_classes}\n{'='*50}\n"
    )
    return num_features, num_classes, train_loader, eval_loader, test_loader


def model_setup(
    args: argparse.Namespace, num_features: int, num_classes: int
) -> nn.Module:
    if args.model == "spikede":
        model = SpikeDEGCN(
            num_features,
            num_classes,
            args.tau,
            args.tau_learnable,
            args.threshold,
            True if args.task == "DRSGCN" else False,
            5,
            "sigmoid_surrogate",
            args.integrator_indicator,
            args.integrator_method,
            args.beta,
            args.time_steps,
            1.0,
            1.0,
            args.dropout,
        ).to(args.device)
    elif args.model == "spikingjelly":
        model = SpikingJellyGCN(
            num_features,
            num_classes,
            args.tau,
            args.threshold,
            True if args.task == "DRSGCN" else False,
            args.time_steps,
            args.dropout,
        ).to(args.device)
    print(f"{'='*50}\nModel Setup:\n{model.eval()}\n{'='*50}\n")
    return model


def args_setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SpikeDE experiment program.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "cora",
            "pubmed",
            "citeseer",
            "amazon_photo",
            "amazon_computers",
            "ogbn-arxiv",
        ],
        help="The name of dataset.",
    )
    parser.add_argument(
        "--task", type=str, required=True, choices=["SGCN", "DRSGCN"], help="Task type."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["spikede", "spikingjelly"],
        help="The type of model.",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        type=float,
        required=False,
        default=[0.7, 0.2, 0.1],
        help="The ratio decides how to split the dataset as train, eval and test.",
    )
    parser.add_argument(
        "--batch_size", type=int, required=False, default=32, help="Batch size."
    )
    parser.add_argument(
        "--positional_method",
        type=str,
        required=False,
        choices=["random_walk", "laplace"],
        default="random_walk",
        help="The method of positional encoding.",
    )
    parser.add_argument(
        "--positional_dim",
        type=int,
        required=False,
        default=16,
        help="The dim of positional code.",
    )
    parser.add_argument(
        "--tau", type=float, required=False, default=2.0, help="The initial tau."
    )
    parser.add_argument(
        "--tau_learnable", action="store_true", help="Whether tau is learnable."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=False,
        default=1.0,
        help="The initial threshold.",
    )
    parser.add_argument(
        "--integrator_indicator",
        type=str,
        required=False,
        choices=["odeint", "fdeint", "odeint_adjoint", "fdeint_adjoint"],
        default="odeint_adjoint",
        help="The integrator indicator of SpikeDE model.",
    )
    parser.add_argument(
        "--integrator_method",
        type=str,
        required=False,
        choices=[
            "euler",
            "predictor-f",
            "predictor-o",
            "trap-f",
            "trap-o",
            "gl-f",
            "gl-o",
            "predictor",
            "implicitl1",
            "gl",
            "trap",
        ],
        default="euler",
        help="The integrator method of SpikeDE model. \
            This parameter is up to integrator indicator: 'odeint' and 'odeint_adjoint' only support 'euler'; \
                'fdeint' only support 'predictor', 'implicitl1', 'gl' and 'trap'; \
                    'fdeint_adjoint' only support 'predictor-f', 'predictor-o', 'trap-f', 'trap-o', 'gl-f' and 'gl-o'.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        required=False,
        default=0.5,
        help="The beta param of SpikeDE model.",
    )
    parser.add_argument(
        "--time_steps",
        type=int,
        required=False,
        default=32,
        help="The total time steps of spiking neural network.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        required=False,
        default=0.6,
        help="The dropout p of models.",
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        required=False,
        default=0.001,
        help="The learning rate of trainig.",
    )
    parser.add_argument(
        "--epochs", type=int, required=False, default="100", help="Training epochs."
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=False,
        default="outputs",
        help="The path to experiment outputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cpu",
        help="Which device to use.",
    )
    parser.add_argument(
        "--edge_keep_ratio",
        type=float,
        required=False,
        default=1.0,
        help="How many edges to be kept.",
    )
    parser.add_argument(
        "--test_dropout",
        type=float,
        required=False,
        default=0.0,
        help="The dropout p of graph features in test stage.",
    )

    args = parser.parse_args()
    return args


def main():
    args = args_setup()
    output_dir = os.path.join(args.output_dir, args.task, args.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_features, num_classes, train_loader, eval_loader, test_loader = task_setup(args)
    model = model_setup(args, num_features, num_classes)
    criterion = nn.CrossEntropyLoss()

    if args.model == "spikingjelly":
        file_prefix = "spikingjelly"
    elif args.model == "spikede":
        file_prefix = (
            "spikede_ode"
            if args.integrator_indicator in ["odeint", "odeint_adjoint"]
            else "spikede_fde"
        )

    model.load_state_dict(
        torch.load(
            os.path.join(output_dir, f"{file_prefix}.pth"),
            map_location=torch.device(args.device),
        )
    )
    test_accs, std = test_model(
        model,
        criterion,
        test_loader,
        True if args.model == "spikingjelly" else False,
        args.test_dropout,
        args.device,
    )
    energy, params = consumption(
        model,
        test_loader,
        True if args.model == "spikingjelly" else False,
        device=args.device,
    )

    results = io.StringIO()
    print(
        f"{'='*50}\nHyper Parameters:\n{'-'*50}\n{pformat(vars(args))}\n{'='*50}\n",
        file=results,
    )
    print(
        f"{'=' * 50}\nTest Results:\n{'-' * 50}\n{'Noise':<20} {'Accuracy (%)':>20}\n{'-' * 50}",
        file=results,
    )
    for noise, acc in test_accs.items():
        if noise == (0, 0):
            print(
                f"{f'mean={noise[0]:.2f}, std={noise[1]:.2f}':<25} {(acc * 100):>10.2f} ± {(std * 100):.2f}",
                file=results,
            )
        else:
            print(
                f"{f'mean={noise[0]:.2f}, std={noise[1]:.2f}':<25} {(acc * 100):>10.2f}%",
                file=results,
            )
    print(f'{"=" * 50}\n', file=results)
    print(
        f"energy consumption: {energy * 1e3:.3e} mJ, number of parameters: {params / 1000:.3f} K",
        file=results,
    )

    with open(os.path.join(output_dir, f"{file_prefix}.txt"), "w") as file:
        file.write(results.getvalue())
    print(results.getvalue())
    results.close()


if __name__ == "__main__":
    main()
