"""MimicBFO
============
A faithful PyTorch re-implementation of the Schwefel-tuned NumPy BFO that uses
interaction-adjusted fitness, cumulative health for reproduction and per-bacterium
step sizes.

The purpose is to benchmark against our more sophisticated ChaoticBFO and locate
which design choices drive Schwefel success-rates.
"""

from typing import Callable, Dict, Any, Optional, Tuple, Iterable, List, Union
import torch
import numpy as np
from torch.optim import Optimizer


def _interaction_cost(positions: torch.Tensor,
                      d_attr: float,
                      w_attr: float,
                      h_rep: float,
                      w_rep: float) -> torch.Tensor:
    """Compute scalar swarming cost J_cc for every bacterium (vectorised)."""
    # positions: (S, D)
    S = positions.shape[0]
    # pairwise squared distances
    dist2 = torch.cdist(positions, positions, p=2) ** 2 + 1e-12  # (S, S)
    attract = -d_attr * torch.exp(-w_attr * dist2)
    repel = h_rep * torch.exp(-w_rep * torch.sqrt(dist2))
    # Exclude self
    mask = ~torch.eye(S, dtype=torch.bool, device=positions.device)
    return (attract + repel)[mask].view(S, S - 1).sum(dim=1)  # (S,)


class MimicBFO(Optimizer):
    def __init__(
        self,
        params: Iterable[Union[torch.Tensor, Dict[str, Any]]],
        lr: float = 0.01,  # unused (kept for API)
        population_size: int = 50,
        chemotaxis_steps: int = 100,
        swim_length: int = 4,
        reproduction_steps: int = 5,
        elimination_steps: int = 4,
        elimination_prob: float = 0.25,
        step_size_init: float = 50.0,
        step_decay: float = 0.95,
        swarming_params: Tuple[float, float, float, float] = (0.1, 0.2, 0.1, 10.0),
        domain_bounds: Tuple[float, float] = (-500.0, 500.0),
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        self.pop_size = population_size
        self.Nc = chemotaxis_steps
        self.Ns = swim_length
        self.Nre = reproduction_steps
        self.Ned = elimination_steps
        self.Ped = elimination_prob
        self.C_init = step_size_init
        self.step_decay = step_decay
        self.d_attr, self.w_attr, self.h_rep, self.w_rep = swarming_params
        self.lower, self.upper = domain_bounds

        # Torch bookkeeping – flatten parameter list into single vector
        params = list(params)
        if len(params) != 1:
            raise ValueError("MimicBFO expects a single torch.nn.Parameter to optimise")
        self._flat_param = params[0]  # nn.Parameter
        if device is None:
            device = self._flat_param.device
        self.device = device

        defaults = dict(lr=lr)
        super().__init__([self._flat_param], defaults)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        D = self._flat_param.data.numel()
        # Initialise population and per-bacterium step sizes C_arr
        self.theta = torch.rand((self.pop_size, D), device=device) * (self.upper - self.lower) + self.lower
        self.C_arr = torch.full((self.pop_size,), self.C_init, device=device)
        # Best seen
        self.best_pos = None
        self.best_cost = float('inf')

    def step(self, closure: Callable[[], float], max_fe: Optional[int] = None):
        """Run a *full* BFO macro-loop (Ned → Nre → Nc chemotaxis)."""
        # (max_fe is ignored; MimicBFO always runs a fixed schedule)
        for l in range(self.Ned):
            for k in range(self.Nre):
                health = torch.zeros(self.pop_size, device=self.device)
                for j in range(self.Nc):
                    # J_all (interaction-adjusted)
                    cost_all = self._evaluate_cost_batch(closure)
                    J_cc = _interaction_cost(self.theta, self.d_attr, self.w_attr,
                                              self.h_rep, self.w_rep)
                    J_all = cost_all + J_cc
                    health += J_all

                    # Tumble
                    delta = torch.rand_like(self.theta) * 2 - 1  # U[-1,1]
                    norm = delta.norm(dim=1, keepdim=True)
                    phi = torch.zeros_like(delta)
                    phi[norm.squeeze() > 0] = delta[norm.squeeze() > 0] / norm[norm.squeeze() > 0]
                    current_pos = self.theta + self.C_arr.view(-1, 1) * phi
                    current_pos.clamp_(self.lower, self.upper)

                    # Evaluate after tumble
                    cost_all = self._evaluate_cost_batch(closure, current_pos)
                    J_cc = _interaction_cost(current_pos, self.d_attr, self.w_attr,
                                              self.h_rep, self.w_rep)
                    J = cost_all + J_cc

                    # Update global best
                    min_idx = torch.argmin(cost_all)
                    if cost_all[min_idx] < self.best_cost:
                        self.best_cost = cost_all[min_idx].item()
                        self.best_pos = current_pos[min_idx].clone()

                    # Swim
                    J_last = J.clone()
                    for m in range(self.Ns):
                        active = J < J_last
                        if not torch.any(active):
                            break
                        health[active] += J[active]
                        J_last[active] = J[active]
                        current_pos[active] += self.C_arr[active].view(-1, 1) * phi[active]
                        current_pos.clamp_(self.lower, self.upper)
                        cost_all = self._evaluate_cost_batch(closure, current_pos)
                        J_cc = _interaction_cost(current_pos, self.d_attr, self.w_attr,
                                                  self.h_rep, self.w_rep)
                        J = cost_all + J_cc
                        min_idx = torch.argmin(cost_all)
                        if cost_all[min_idx] < self.best_cost:
                            self.best_cost = cost_all[min_idx].item()
                            self.best_pos = current_pos[min_idx].clone()
                    # Update theta for next chemotaxis step
                    self.theta = current_pos
                # Reproduction
                sort_idx = torch.argsort(health)
                self.theta = self.theta[sort_idx]
                self.C_arr = self.C_arr[sort_idx]
                Sr = self.pop_size // 2
                self.theta[Sr:] = self.theta[:Sr].clone()
                self.C_arr[Sr:] = self.C_arr[:Sr]
                # Step size decay
                self.C_arr *= self.step_decay
            # Elimination–dispersal
            mask = torch.rand(self.pop_size, device=self.device) < self.Ped
            num = mask.sum().item()
            if num > 0:
                self.theta[mask] = torch.rand(num, self.theta.shape[1], device=self.device) * (self.upper - self.lower) + self.lower
                self.C_arr[mask] = self.C_init

        # After full loop, copy best parameters back to model
        with torch.no_grad():
            self._flat_param.data.copy_(self.best_pos)
        return self.best_cost

    # ---------------------------------------------------------------------
    def _evaluate_cost_batch(self, closure: Callable[[], float], positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Evaluate objective for each bacterium; closure expects model params already set."""
        if positions is None:
            positions = self.theta
        costs = torch.empty(self.pop_size, device=self.device)
        for i in range(self.pop_size):
            with torch.no_grad():
                self._flat_param.data.copy_(positions[i])
            costs[i] = closure()
        return costs 