import torch, torch.nn as nn
from torch.functional import F
from fspikeDE import snn, LIFNeuron
from spikingjelly.clock_driven import neuron, encoding


class SpikeDEGCN(nn.Module):
    """
    SpikingGCN using our SpikeDE package.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau: float = 10.0,
        tau_learnable: bool = False,
        threshold: float = 1.0,
        threshold_learnable: bool = False,
        surrogate_grad_scale: float = 5.0,
        surrogate_opt: str = "sigmoid_surrogate",
        integrator_indicator: str = "odeint_adjoint",
        integrator_method: str = "euler",
        beta: float = 0.5,
        time_steps: int = 32,
        time_interval: float = 1.0,
        step_size: float = 1.0,
        dropout: float = 0.6,
    ) -> None:
        super(SpikeDEGCN, self).__init__()
        assert (
            step_size <= time_interval
        ), "step_size must be less than or equal to time_interval."
        self.flops = in_features * out_features
        self.time_steps = time_steps
        self.time_interval = time_interval
        self.step_size = step_size
        self.integrator_method = integrator_method
        self.threshold = (
            nn.Parameter(
                torch.tensor(threshold, dtype=torch.float32), requires_grad=True
            )
            if threshold_learnable
            else threshold
        )

        sequential = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Dropout(dropout),
            LIFNeuron(
                tau,
                self.threshold,
                surrogate_grad_scale,
                surrogate_opt,
                tau_learnable,
            ),
        )

        self.net = snn(sequential, "snn", integrator=integrator_indicator, beta=beta)
        self.encoder = encoding.PoissonEncoder()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        spikes = torch.full(
            (self.time_steps, inputs.shape[0], inputs.shape[1]),
            0,
            dtype=torch.float32,
            device=inputs.device,
        )
        for t in range(self.time_steps):
            spikes[t] = self.encoder(inputs)

        inputs_time = torch.linspace(
            0,
            self.time_interval * (self.time_steps - 1),
            self.time_steps,
            device=inputs.device,
        )

        options = {"step_size": self.step_size}
        v_mem_all_time_and_cumulated_spike = self.net(
            spikes,
            inputs_time,
            method=self.integrator_method,
            options=options,
        )

        cumulated_spike = v_mem_all_time_and_cumulated_spike[-1][-1]
        spike_freq = F.softmax(cumulated_spike, dim=1)
        return spike_freq


class SpikingJellyGCN(nn.Module):
    """
    SpikingGCN using our SpikingJelly.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau: float = 10.0,
        threshold: float = 1.0,
        threshold_learnable: bool = False,
        time_steps: int = 32,
        dropout: float = 0.6,
    ) -> None:
        super(SpikingJellyGCN, self).__init__()
        self.time_steps = time_steps
        self.flops = in_features * out_features
        self.threshold = (
            nn.Parameter(
                torch.tensor(threshold, dtype=torch.float32), requires_grad=True
            )
            if threshold_learnable
            else threshold
        )

        self.sequential = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Dropout(dropout),
            neuron.LIFNode(tau, False),
        )
        del self.sequential[-1].v_threshold
        self.sequential[-1].v_threshold = self.threshold

        self.encoder = encoding.PoissonEncoder()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for t in range(self.time_steps):
            inputs = self.encoder(inputs)
            if t == 0:
                out_spikes = self.sequential(inputs)
            else:
                out_spikes += self.sequential(inputs)

        return F.softmax(out_spikes, dim=1)
