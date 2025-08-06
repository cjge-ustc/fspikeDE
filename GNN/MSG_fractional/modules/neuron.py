import numpy as np
import torch
import torch.nn as nn
import math

from torch.xpu import device


class IFFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, manifold, x_seq, v_seq, z_seq, v_threshold=1.0, method="gl"):
        ctx.manifold = manifold
        ctx.save_for_backward(x_seq, v_seq, z_seq)
        men_potential = torch.zeros_like(v_seq).to(v_seq.device)
        s_seq = []

        # for t in range(x_seq.shape[0]):
        #     men_potential = men_potential + x_seq[t]
        #     spike = (men_potential > v_threshold).float()
        #     s_seq.append(spike)
        #     men_potential = men_potential - v_threshold * spike

        print("--------IF--------", method)

        N = len(x_seq)
        alpha = 0.5  ###fde order beta
        device = x_seq.device
        h = torch.tensor(1.0, device=device)  ### step size

        if method == "gl":
            c = torch.zeros(N + 1, dtype=torch.float64, device=device)
            c[0] = 1
            for j in range(1, N + 1):
                c[j] = (1 - (1 + alpha) / j) * c[j - 1]

            y_history = []
            y_history.append(men_potential)
            for k in range(1, N + 1):
                right = 0
                for j in range(1, k + 1):
                    right = right + c[j] * y_history[k - j]
                men_potential = x_seq[k - 1] * torch.pow(h, alpha) - right

                spike = (men_potential > v_threshold).float()
                s_seq.append(spike)
                men_potential = men_potential - v_threshold * spike
                y_history.append(men_potential)

        elif method == "predictor":
            gamma_alpha = 1 / math.gamma(alpha)
            fhistory = []
            spike = 0
            for k in range(N):
                f_k = x_seq[k] - v_threshold * spike
                fhistory.append(f_k)

                memory_k = 0
                j_vals = torch.arange(
                    0, k + 1, dtype=torch.float32, device=device
                ).unsqueeze(1)
                b_j_k_1 = (torch.pow(h, alpha) / alpha) * (
                    torch.pow(k + 1 - j_vals, alpha) - torch.pow(k - j_vals, alpha)
                )
                b_all_k = torch.zeros_like(
                    fhistory[memory_k]
                )  # Initialize accumulator with zeros of the same shape as the product

                # Loop through the range and accumulate results
                for i in range(memory_k, k + 1):
                    b_all_k += b_j_k_1[i] * fhistory[i]
                # temp_product = torch.stack([b_j_k_1[i] * fhistory[i] for i in range(memory_k, k + 1)])
                # b_all_k = torch.sum(temp_product, dim=0)

                men_potential = gamma_alpha * b_all_k

                spike = (men_potential > v_threshold).float()
                s_seq.append(spike)

        output = torch.stack(s_seq, dim=0)
        z_output = manifold.expmap(z_seq, v_seq)
        return output, z_output

    @staticmethod
    def backward(ctx, grad_output, grad_z_output):
        # print(f"grad_output: {grad_output.norm(p=2)}, grad_z_output: {grad_z_output.norm(p=2)}")
        x_seq, v_seq, z_seq = ctx.saved_tensors
        manifold = ctx.manifold
        jacob_v = manifold.jacobian_expmap_v(z_seq, v_seq)
        jacob_x = manifold.jacobian_expmap_x(z_seq, v_seq)
        # print(f"jacob_v: {jacob_v.norm(p=2)}, jacob_x: {jacob_x.norm(p=2)}")
        # print(f"jacob_x: {jacob_v.shape}, grad_z_output: {grad_z_output.shape}")
        grad_v = jacob_v.transpose(-1, -2) @ grad_z_output.unsqueeze(-1)
        grad_z = jacob_x.transpose(-1, -2) @ grad_z_output.unsqueeze(-1)
        # print(f"grad_v: {grad_v.norm(p=2)}, grad_z: {grad_z.norm(p=2)}")
        return None, None, grad_v.squeeze(), grad_z.squeeze(), None, None


class RiemannianIFNode(nn.Module):
    def __init__(
        self, manifold, v_threshold: float = 1.0, delta=0.05, tau=2, method="gl"
    ):
        super(RiemannianIFNode, self).__init__()
        self.manifold = manifold
        self.v_threshold = v_threshold
        self.method = method

    def forward(self, x_seq: torch.Tensor, v_seq: torch.Tensor, z_seq: torch.Tensor):
        """

        :param x_seq: [T, N, D]
        :param v_seq: [N, D]
        :param z_seq: [N, D]
        :return:
        """
        out_seq, z_out_seq = IFFunction.apply(
            self.manifold, x_seq, v_seq, z_seq, self.v_threshold, self.method
        )
        return out_seq, z_out_seq


# class LIFFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, manifold, x_seq, v_seq, z_seq, v_threshold=1.0, delta_t=0.05, tau=2.):
#         ctx.manifold = manifold
#         ctx.save_for_backward(x_seq, v_seq, z_seq)
#         men_potential = torch.zeros_like(v_seq).to(v_seq.device)
#         s_seq = []
#         beta = np.exp(-delta_t / tau)
#         for t in range(x_seq.shape[0]):
#             men_potential = beta * men_potential + (1 - beta) * x_seq[t]
#             spike = (men_potential > v_threshold).float()
#             s_seq.append(spike)
#             men_potential = men_potential - v_threshold * spike
#
#         output = torch.stack(s_seq, dim=0)
#         z_output = manifold.expmap(z_seq, v_seq)
#         return output, z_output


class LIFFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        manifold,
        x_seq,
        v_seq,
        z_seq,
        v_threshold=1.0,
        delta_t=0.05,
        tau=2.0,
        method="gl",
    ):
        ctx.manifold = manifold
        ctx.save_for_backward(x_seq, v_seq, z_seq)
        men_potential = torch.zeros_like(v_seq).to(v_seq.device)
        s_seq = []

        # beta = np.exp(-delta_t / tau)
        # for t in range(x_seq.shape[0]):
        #     men_potential = beta * men_potential + (1 - beta) * x_seq[t]
        #     spike = (men_potential > v_threshold).float()
        #     s_seq.append(spike)
        #     men_potential = men_potential - v_threshold * spike

        print("--------LIF--------", method)

        N = len(x_seq)
        h = delta_t
        alpha = 0.5
        device = x_seq.device
        h = torch.tensor(h, device=device)
        if method == "gl":
            c = torch.zeros(N + 1, dtype=torch.float64, device=device)
            c[0] = 1
            for j in range(1, N + 1):
                c[j] = (1 - (1 + alpha) / j) * c[j - 1]

            y_history = []
            y_history.append(men_potential)
            for k in range(1, N + 1):

                spike = (men_potential > v_threshold).float()
                s_seq.append(spike)

                right = 0
                for j in range(1, k + 1):
                    right = right + c[j] * y_history[k - j]
                f_k = 1 / tau * (-men_potential + x_seq[k - 1] - v_threshold * spike)
                men_potential = f_k * torch.pow(h, alpha) - right

                # men_potential = men_potential - v_threshold * spike
                y_history.append(men_potential)

            # pop the first element of s_seq
            s_seq.pop(0)
            spike = (men_potential > v_threshold).float()
            s_seq.append(spike)

        elif method == "predictor":
            gamma_alpha = 1 / math.gamma(alpha)
            fhistory = []
            spike = 0
            for k in range(N):
                f_k = 1 / tau * (-men_potential + x_seq[k] - v_threshold * spike)
                fhistory.append(f_k)

                memory_k = 0
                j_vals = torch.arange(
                    0, k + 1, dtype=torch.float32, device=device
                ).unsqueeze(1)
                b_j_k_1 = (torch.pow(h, alpha) / alpha) * (
                    torch.pow(k + 1 - j_vals, alpha) - torch.pow(k - j_vals, alpha)
                )
                b_all_k = torch.zeros_like(
                    fhistory[memory_k]
                )  # Initialize accumulator with zeros of the same shape as the product

                # Loop through the range and accumulate results
                for i in range(memory_k, k + 1):
                    b_all_k += b_j_k_1[i] * fhistory[i]
                # temp_product = torch.stack([b_j_k_1[i] * fhistory[i] for i in range(memory_k, k + 1)])
                # b_all_k = torch.sum(temp_product, dim=0)

                men_potential = gamma_alpha * b_all_k

                spike = (men_potential > v_threshold).float()
                s_seq.append(spike)

        output = torch.stack(s_seq, dim=0)
        z_output = manifold.expmap(z_seq, v_seq)
        return output, z_output

    @staticmethod
    def backward(ctx, grad_output, grad_z_output):
        # print(f"grad_output: {grad_output.norm(p=2)}, grad_z_output: {grad_z_output.norm(p=2)}")
        x_seq, v_seq, z_seq = ctx.saved_tensors
        manifold = ctx.manifold
        jacob_v = manifold.jacobian_expmap_v(z_seq, v_seq)
        jacob_x = manifold.jacobian_expmap_x(z_seq, v_seq)
        # print(f"jacob_v: {jacob_v.norm(p=2)}, jacob_x: {jacob_x.norm(p=2)}")
        # print(f"jacob_x: {jacob_v.shape}, grad_z_output: {grad_z_output.shape}")
        grad_v = jacob_v.transpose(-1, -2) @ grad_z_output.unsqueeze(-1)
        grad_z = jacob_x.transpose(-1, -2) @ grad_z_output.unsqueeze(-1)
        # print(f"grad_v: {grad_v.norm(p=2)}, grad_z: {grad_z.norm(p=2)}")
        return None, None, grad_v.squeeze(), grad_z.squeeze(), None, None, None, None


class RiemannianLIFNode(nn.Module):
    def __init__(
        self, manifold, v_threshold: float = 1.0, delta=0.05, tau=2.0, method="gl"
    ):
        super(RiemannianLIFNode, self).__init__()
        self.manifold = manifold
        self.v_threshold = v_threshold
        self.delta = delta
        self.tau = tau
        self.method = method

    def forward(self, x_seq: torch.Tensor, v_seq: torch.Tensor, z_seq: torch.Tensor):
        """

        :param x_seq: [T, N, D]
        :param v_seq: [N, D]
        :param z_seq: [N, D]
        :return:
        """
        out_seq, z_out_seq = LIFFunction.apply(
            self.manifold,
            x_seq,
            v_seq,
            z_seq,
            self.v_threshold,
            self.delta,
            self.tau,
            self.method,
        )
        return out_seq, z_out_seq


RiemannianNeuron = {"IF": RiemannianIFNode, "LIF": RiemannianLIFNode}

from spikingjelly.clock_driven.neuron import MultiStepIFNode, MultiStepLIFNode


class IFNode(nn.Module):
    def __init__(self, manifold, v_threshold: float = 1.0, delta=0.05, tau=2):
        super(IFNode, self).__init__()
        self.manifold = manifold
        self.neuron = MultiStepIFNode(v_threshold=v_threshold, detach_reset=True)
        print("Using IF Node")

    def forward(self, x_seq: torch.Tensor, v_seq: torch.Tensor, z_seq: torch.Tensor):
        """

        :param x_seq: [T, N, D]
        :param v_seq: [N, D]
        :param z_seq: [N, D]
        :return:
        """
        out_seq = self.neuron(x_seq)
        z_out_seq = self.manifold.expmap(z_seq, v_seq)
        return out_seq, z_out_seq


class LIFNode(nn.Module):
    def __init__(self, manifold, v_threshold: float = 1.0, delta=0.05, tau=2.0):
        super(LIFNode, self).__init__()
        self.manifold = manifold
        self.neuron = MultiStepLIFNode(
            v_threshold=v_threshold, detach_reset=True, tau=tau
        )
        print("Using LIF Node")

    def forward(self, x_seq: torch.Tensor, v_seq: torch.Tensor, z_seq: torch.Tensor):
        """

        :param x_seq: [T, N, D]
        :param v_seq: [N, D]
        :param z_seq: [N, D]
        :return:
        """
        out_seq = self.neuron(x_seq)
        z_out_seq = self.manifold.expmap(z_seq, v_seq)
        return out_seq, z_out_seq


Neuron = {"IF": IFNode, "LIF": LIFNode}
