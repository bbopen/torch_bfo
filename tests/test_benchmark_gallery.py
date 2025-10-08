import math

import pytest
import torch
import torch.nn as nn

from bfo_torch import BFO, AdaptiveBFO, HybridBFO


def run_optimizer(fn, bounds, dim, steps=20, pop=10):
    torch.manual_seed(0)
    x = nn.Parameter(torch.empty(dim).uniform_(bounds[0], bounds[1]))
    optimizer = BFO(
        [x],
        lr=0.05,
        population_size=pop,
        chemotaxis_steps=2,
        swim_length=2,
        reproduction_steps=1,
        elimination_steps=1,
        step_size_max=0.1 * (bounds[1] - bounds[0]),
        domain_bounds=bounds,
        seed=0,
        early_stopping=False,
    )

    def closure():
        return fn(x).item()

    best = closure()
    for _ in range(steps):
        best = optimizer.step(closure)
    return best


# ---- Benchmark functions ----


def rastrigin(x):
    n = x.numel()
    return 10 * n + torch.sum(x**2 - 10 * torch.cos(2 * math.pi * x))


def ackley(x):
    n = x.numel()
    return (
        -20 * torch.exp(-0.2 * torch.sqrt((x**2).mean()))
        - torch.exp(torch.cos(2 * math.pi * x).mean())
        + 20
        + math.e
    )


def griewank(x):
    part1 = torch.sum(x**2) / 4000
    i = torch.arange(1, x.numel() + 1, device=x.device, dtype=x.dtype)
    part2 = torch.prod(torch.cos(x / torch.sqrt(i)))
    return part1 - part2 + 1


def schwefel(x):
    n = x.numel()
    return 418.9829 * n - torch.sum(x * torch.sin(torch.sqrt(torch.abs(x))))


def weierstrass(x, a=0.5, b=3, k_max=20):
    terms = [a**k * torch.cos(2 * math.pi * b**k * (x + 0.5)) for k in range(k_max + 1)]
    term0 = sum(terms)
    const = sum(
        a**k * torch.cos(torch.tensor(math.pi * b**k)) for k in range(k_max + 1)
    )
    return term0.sum() - x.numel() * const


def katsuura(x):
    i = torch.arange(1, x.numel() + 1, device=x.device, dtype=x.dtype)
    prod = 1.0
    for xi, ii in zip(x, i):
        abs_sum = torch.sum(
            torch.abs(
                2 ** torch.arange(1, 33, device=x.device) * xi
                - torch.round(2 ** torch.arange(1, 33, device=x.device) * xi)
            )
            / 2 ** torch.arange(1, 33, device=x.device)
        )
        prod *= (1 + ii * abs_sum) ** (10 / x.numel() ** 1.2)
    return prod


def lunacek(x):
    n = x.numel()
    mu1 = 2.5
    d = 1.0
    s = 1 - 1 / (2 * math.sqrt(n + 20) - 8.2)
    mu2 = -math.sqrt((mu1**2 - d) / s)
    first = torch.sum((x - mu1) ** 2)
    second = d * n + s * torch.sum((x - mu2) ** 2)
    r = torch.min(first, second)
    rastrigin_term = 10 * torch.sum(1 - torch.cos(2 * math.pi * (x - mu1)))
    return r + rastrigin_term


def michalewicz(x, m=10):
    i = torch.arange(1, x.numel() + 1, device=x.device, dtype=x.dtype)
    return -torch.sum(torch.sin(x) * torch.sin(i * x**2 / math.pi) ** (2 * m))


def schubert(x):
    x1, x2 = x[0], x[1]
    sum1 = torch.sum(

            torch.arange(1, 6, device=x.device)
            * torch.cos(
                (torch.arange(1, 6, device=x.device) + 1) * x1
                + torch.arange(1, 6, device=x.device)
            )

    )
    sum2 = torch.sum(

            torch.arange(1, 6, device=x.device)
            * torch.cos(
                (torch.arange(1, 6, device=x.device) + 1) * x2
                + torch.arange(1, 6, device=x.device)
            )

    )
    return sum1 * sum2


def levy(x):
    w = 1 + (x - 1) / 4
    term1 = torch.sin(math.pi * w[0]) ** 2
    term2 = ((w[:-1] - 1) ** 2 * (1 + 10 * torch.sin(math.pi * w[:-1] + 1) ** 2)).sum()
    term3 = (w[-1] - 1) ** 2 * (1 + torch.sin(2 * math.pi * w[-1]) ** 2)
    return term1 + term2 + term3


def easom(x):
    return (
        -torch.cos(x[0])
        * torch.cos(x[1])
        * torch.exp(-((x[0] - math.pi) ** 2 + (x[1] - math.pi) ** 2))
    )


def rosenbrock(x):
    return torch.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)


def step(x):
    return torch.sum(torch.floor(x + 0.5) ** 2)


def quartic_noise(x):
    i = torch.arange(1, x.numel() + 1, device=x.device, dtype=x.dtype)
    return torch.sum(i * x**4) + torch.rand(1).item()


def drop_wave(x):
    r2 = torch.sum(x**2)
    return -(1 + torch.cos(12 * torch.sqrt(r2))) / (0.5 * r2 + 2)


# ---- Tests ----


@pytest.mark.parametrize(
    "fn,bounds,dim",
    [
        (rastrigin, (-5.12, 5.12), 5),
        (ackley, (-32.768, 32.768), 5),
        (griewank, (-600, 600), 5),
        (schwefel, (-500, 500), 5),
        (weierstrass, (-0.5, 0.5), 5),
        (katsuura, (-100, 100), 5),
        (lunacek, (-5, 5), 5),
        (michalewicz, (0, math.pi), 5),
        (schubert, (-10, 10), 2),
        (levy, (-10, 10), 5),
        (easom, (-100, 100), 2),
        (rosenbrock, (-30, 30), 5),
        (step, (-100, 100), 5),
        (quartic_noise, (-1.28, 1.28), 5),
        (drop_wave, (-5.12, 5.12), 2),
    ],
)
def test_benchmark_function(fn, bounds, dim):
    torch.manual_seed(0)
    x = nn.Parameter(torch.empty(dim).uniform_(bounds[0], bounds[1]))
    initial_loss = fn(x).item()
    final_loss = run_optimizer(fn, bounds, dim)
    assert torch.isfinite(torch.tensor(final_loss))
    assert final_loss <= initial_loss


# ---- Convergence Tests with Criteria ----


def sphere(x):
    """Simple unimodal function: f(x) = Σx²ᵢ, optimum at 0."""
    return (x ** 2).sum()


def test_sphere_convergence():
    """Sphere function should converge to near-zero."""
    torch.manual_seed(42)
    dim = 10
    x = nn.Parameter(torch.randn(dim) * 2)  # Start around [-2, 2]
    
    optimizer = BFO(
        [x],
        lr=0.1,
        population_size=30,
        chemotaxis_steps=5,
        swim_length=4,
        reproduction_steps=2,
        elimination_steps=2,
        step_size_max=0.5,
        domain_bounds=(-5.0, 5.0),
        seed=42,
    )
    
    def closure():
        return sphere(x).item()
    
    initial_loss = closure()
    
    # Run for reasonable number of steps
    for _ in range(10):
        final_loss = optimizer.step(closure)
    
    # Should converge well on this simple function
    assert final_loss < initial_loss * 0.2, f"Sphere: Expected <{initial_loss * 0.2:.3e}, got {final_loss:.3e}"
    assert final_loss < 1.0, f"Sphere: Expected <1.0, got {final_loss:.3e}"


def test_rosenbrock_convergence():
    """Rosenbrock (narrow valley) should reach reasonable value."""
    torch.manual_seed(42)
    x = nn.Parameter(torch.tensor([0.0, 0.0]))  # Start at origin
    
    optimizer = BFO(
        [x],
        lr=0.1,
        population_size=50,
        chemotaxis_steps=5,
        swim_length=4,
        reproduction_steps=3,
        elimination_steps=2,
        step_size_max=0.3,
        domain_bounds=(-2.0, 2.0),
        seed=42,
    )
    
    def closure():
        return rosenbrock(x).item()
    
    initial_loss = closure()
    
    # Run longer for harder function
    for _ in range(15):
        final_loss = optimizer.step(closure)
    
    # Should make significant progress
    assert final_loss < initial_loss * 0.5, f"Rosenbrock: Expected <{initial_loss * 0.5:.3f}, got {final_loss:.3f}"
    assert final_loss < 10.0, f"Rosenbrock: Expected <10.0, got {final_loss:.3f}"


def test_rastrigin_convergence():
    """Rastrigin (highly multimodal) should find reasonable local minimum."""
    torch.manual_seed(42)
    dim = 5
    x = nn.Parameter(torch.randn(dim) * 2)
    
    optimizer = BFO(
        [x],
        lr=0.1,
        population_size=50,
        chemotaxis_steps=5,
        swim_length=4,
        reproduction_steps=2,
        elimination_steps=3,  # More elimination for multimodal
        elimination_prob=0.3,
        step_size_max=0.5,
        levy_alpha=1.5,  # Lévy flights help with multimodal
        domain_bounds=(-5.12, 5.12),
        seed=42,
    )
    
    def closure():
        return rastrigin(x).item()
    
    initial_loss = closure()
    
    # Run for many steps (multimodal is hard)
    for _ in range(20):
        final_loss = optimizer.step(closure)
    
    # Should find a decent local minimum
    assert final_loss < initial_loss * 0.7, f"Rastrigin: Expected <{initial_loss * 0.7:.1f}, got {final_loss:.1f}"
    # Rastrigin has many local minima, so just check finite and improved
    assert torch.isfinite(torch.tensor(final_loss))


def test_ackley_convergence():
    """Ackley (deep local minima) should converge reasonably."""
    torch.manual_seed(42)
    dim = 5
    x = nn.Parameter(torch.randn(dim) * 5)
    
    optimizer = BFO(
        [x],
        lr=0.1,
        population_size=40,
        chemotaxis_steps=5,
        swim_length=3,
        reproduction_steps=2,
        elimination_steps=2,
        step_size_max=1.0,
        domain_bounds=(-10.0, 10.0),
        seed=42,
    )
    
    def closure():
        return ackley(x).item()
    
    initial_loss = closure()
    
    for _ in range(15):
        final_loss = optimizer.step(closure)
    
    # Ackley has global optimum at 0
    assert final_loss < initial_loss * 0.5, f"Ackley: Expected <{initial_loss * 0.5:.2f}, got {final_loss:.2f}"
    assert final_loss < 5.0, f"Ackley: Expected <5.0, got {final_loss:.2f}"


# ---- Test All Three Variants ----


@pytest.mark.parametrize("optimizer_class", [BFO, AdaptiveBFO, HybridBFO])
def test_all_variants_on_sphere(optimizer_class):
    """All three optimizer variants should work on Sphere function."""
    torch.manual_seed(42)
    x = nn.Parameter(torch.randn(10) * 2)
    
    if optimizer_class == HybridBFO:
        # HybridBFO benefits from gradients
        optimizer = optimizer_class(
            [x],
            lr=0.1,
            population_size=20,
            chemotaxis_steps=3,
            reproduction_steps=1,
            elimination_steps=1,
            seed=42,
        )
        
        def closure():
            optimizer.zero_grad()
            loss = sphere(x)
            loss.backward()
            return loss.item()
    else:
        optimizer = optimizer_class(
            [x],
            lr=0.1,
            population_size=20,
            chemotaxis_steps=3,
            reproduction_steps=1,
            elimination_steps=1,
            seed=42,
        )
        
        def closure():
            return sphere(x).item()
    
    initial_loss = closure()
    
    for _ in range(5):
        final_loss = optimizer.step(closure)
    
    # All variants should improve
    assert final_loss < initial_loss, f"{optimizer_class.__name__} did not improve: {initial_loss:.3e} -> {final_loss:.3e}"
    assert torch.isfinite(torch.tensor(final_loss))


@pytest.mark.parametrize("optimizer_class", [BFO, AdaptiveBFO, HybridBFO])
def test_all_variants_on_rosenbrock(optimizer_class):
    """All three optimizer variants should work on Rosenbrock function."""
    torch.manual_seed(42)
    x = nn.Parameter(torch.tensor([0.0, 0.0]))
    
    if optimizer_class == HybridBFO:
        optimizer = optimizer_class(
            [x],
            lr=0.1,
            population_size=30,
            chemotaxis_steps=3,
            reproduction_steps=2,
            elimination_steps=1,
            seed=42,
        )
        
        def closure():
            optimizer.zero_grad()
            loss = rosenbrock(x)
            loss.backward()
            return loss.item()
    else:
        optimizer = optimizer_class(
            [x],
            lr=0.1,
            population_size=30,
            chemotaxis_steps=3,
            reproduction_steps=2,
            elimination_steps=1,
            seed=42,
        )
        
        def closure():
            return rosenbrock(x).item()
    
    initial_loss = closure()
    
    for _ in range(10):
        final_loss = optimizer.step(closure)
    
    # All variants should improve
    assert final_loss < initial_loss, f"{optimizer_class.__name__} did not improve on Rosenbrock"
    assert torch.isfinite(torch.tensor(final_loss))
