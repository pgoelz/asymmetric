from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from math import ceil, isinf, inf, isnan, log, log1p
from time import time
from typing import Iterable, List, Tuple, Optional, Dict, Callable

import cvxpy as cp
from gurobipy import Model, quicksum, GRB, setParam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import scipy.integrate as integrate
from scipy.special import betainc, beta
import scipy.stats as stats
from seaborn import color_palette


EPS = 1e-9


def jeffreys_interval(trials: int, successes: int, confidence: float) -> Tuple[float, float]:
    """Computes a Jeffrey's confidence interval for a binomial proportion.
    See https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Jeffreys_interval .
    """
    assert 0 < confidence < 1
    alpha = 1 - confidence

    lower = stats.beta.ppf(alpha / 2., .5 + successes, .5 + trials - successes)
    upper = stats.beta.ppf(1. - alpha / 2., .5 + successes, .5 + trials - successes)

    if successes == 0:
        lower = 0.
    elif successes == trials:
        upper = 1.

    return (lower, upper)


class Dist(ABC):
    """Abstract class representing a utility distribution.
    Each distribution has a PDF, a CDF, a function for generating a vector of samples, and a minimum and maximum of its
    range.
    """
    @abstractmethod
    def pdf(self, x: float) -> float:
        pass

    @abstractmethod
    def cdf(self, x: float) -> float:
        # should be defined for x = inf
        pass

    @abstractmethod
    def sample(self, num: int) -> Iterable[float]:
        pass

    @property
    @abstractmethod
    def min(self) -> float:
        pass

    @property
    @abstractmethod
    def max(self) -> float:
        pass


class Uniform(Dist):
    def __init__(self, left_endpoint):
        assert 0 <= left_endpoint < 1
        self.left_endpoint = left_endpoint

    def pdf(self, x):
        if self.left_endpoint <= x <= 1.:
            return 1. / (1. - self.left_endpoint)
        else:
            return 0.

    def cdf(self, x):
        if x < self.left_endpoint:
            return 0.
        elif x <= 1.:
            return (x - self.left_endpoint) / (1. - self.left_endpoint)
        else:
            return 1.

    def sample(self, num):
        return ((1 - self.left_endpoint) * np.random.rand(num)) + self.left_endpoint

    @property
    def min(self):
        return self.left_endpoint

    @property
    def max(self):
        return 1.

    def __str__(self):
        return f"Uniform({self.left_endpoint})"


class PiecewiseLinear(Dist):
    def __init__(self, a: float, base: float = .5):
        self.a = a
        self.base = base
        self.ceiling = 2. - base

    def pdf(self, x) -> float:
        if x > 1. or x < 0.:
            return 0.
        elif x < self.a:
            return self.base + (self.ceiling - self.base) * x / self.a
        else:
            return self.base + (self.ceiling - self.base) * (1. - x) / (1. - self.a)

    def cdf(self, x) -> float:
        if x <= 0.:
            return 0.
        elif x >= 1.:
            return 1.
        elif x < self.a:
            return x * self.base + .5 * (self.ceiling - self.base) / self.a * x**2
        else:
            return 1. - (1. - x) * self.base - .5 * (self.ceiling - self.base) / (1. - self.a) * (1. - x)**2

    def sample(self, num: int) -> Iterable[float]:
        rd = np.random.rand(num, 3)
        x = np.where(rd[:, 0] < self.base,
                     rd[:, 1],
                     np.where(rd[:, 1] < self.a,
                              self.a * np.sqrt(rd[:, 2]),
                              1. - (1. - self.a) * np.sqrt(rd[:, 2])))
        return x

    @property
    def min(self) -> float:
        return 0.

    @property
    def max(self) -> float:
        return 1.

    def __str__(self):
        return f"PiecewiseLinear({self.a}, {self.base})"


class Beta(Dist):
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def pdf(self, x) -> float:
        # this is much faster than stats.beta.pdf(x, self.a, self.b)
        if x > 1.:
            return 0.
        return x**(self.a - 1) * (1-x)**(self.b - 1) /  beta(self.a, self.b)

    def cdf(self, x) -> float:
        # this is much faster than stats.beta.cdf(x, self.a, self.b)
        if x >= 1.:
            return 1.
        new = betainc(self.a, self.b, x)
        assert not isnan(new)
        return new

    def sample(self, num: int) -> Iterable[float]:
        return stats.beta.rvs(self.a, self.b, size=num)

    @property
    def min(self) -> float:
        return 0.

    @property
    def max(self) -> float:
        return 1.

    def __str__(self):
        return f"Beta({self.a}, {self.b})"


def _equalization_multipliers(distributions: List[Dist], delta: float, q: float) -> List[float]:
    n = len(distributions)
    epsilon = delta / (2 * q)
    zs = [0 for _ in range(n)]

    def local_eval(x: float, i: int) -> float:
        z1 = zs[i]
        prod = distributions[i].pdf(x)
        for i2, (dist2, z2) in enumerate(zip(distributions, zs)):
            if i2 != i:
                prod *= dist2.cdf((1 + epsilon) ** (z1 - z2) * x)
        return prod

    while True:
        probs = []
        for i in range(n):
            dist = distributions[i]
            probs.append(integrate.quad(partial(local_eval, i=i), dist.min, dist.max)[0])
        if all(abs(p - 1/n) <= delta for p in probs):
            break
        for i in range(n):
            if probs[i] <= 1/n:
                zs[i] += 1

    return [(1 + epsilon)**z for z in zs]


def _equalization_multipliers_refine(distributions: List[Dist], delta: float, q: float) -> List[float]:
    n = len(distributions)
    mults = [1. for _ in range(n)]

    def local_eval(x: float, i: int) -> float:
        mult1 = mults[i]
        prod = distributions[i].pdf(x)
        for i2, (dist2, mult2) in enumerate(zip(distributions, mults)):
            if i2 != i:
                prod *= dist2.cdf(mult1 / mult2 * x)
        return prod

    working_delta = 1.
    while working_delta > delta:
        working_delta /= 2.
        print(working_delta)
        epsilon = working_delta / (2 * q)

        while True:
            probs = []
            for i in range(n):
                dist = distributions[i]
                probs.append(integrate.quad(partial(local_eval, i=i), dist.min, dist.max)[0])
            if all(abs(p - 1/n) <= working_delta for p in probs):
                break
            for i in range(n):
                if probs[i] <= 1/n:
                    mults[i] *= (1 + epsilon)

    return mults


def round_robin_allocation(utilities: np.ndarray) -> np.ndarray:
    n, m = utilities.shape
    prefs = np.argsort(utilities)  # prefs[i,m-x] gives the item that has the x'th highest utility for i
    allocation = np.full(m, -1)
    next_index = np.full(n, m-1)
    for round in range(m):
        turn = round % n
        index = next_index[turn]
        while allocation[prefs[turn, index]] != -1:
            index -= 1
        allocation[prefs[turn, index]] = turn
        next_index[turn] = index - 1
    return allocation


def multiplier_allocation(utilities: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    n, m = utilities.shape
    assert multipliers.shape == (n,)
    scaled_utilities = utilities * multipliers[:, None]  # multiply each row by multiplier
    return np.argmax(scaled_utilities, axis=0)


def fractional_mnw_allocation(utilities: np.ndarray) -> np.ndarray:
    n, m = utilities.shape
    alloc = cp.Variable(shape=(n, m))
    constraints = [alloc >= 0, alloc <= 1, np.ones(n) @ alloc == 1]
    objective = cp.Maximize(cp.sum(cp.log(cp.multiply(alloc, utilities) @ np.ones(m))))
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return np.argmax(alloc.value, axis=0)


def integer_mnw_allocation(utilities: np.ndarray) -> np.ndarray:
    n, m = utilities.shape
    model = pyo.ConcreteModel()
    model.N = pyo.RangeSet(n)
    model.M = pyo.RangeSet(m)
    model.alloc = pyo.Var(model.N, model.M, within=pyo.Binary)
    model.allocate_once = pyo.ConstraintList()
    for j in model.M:
        model.allocate_once.add(sum(model.alloc[i, j] for i in model.N) == 1)
    model.obj = pyo.Objective(expr=sum(pyo.log(sum(utilities[(iint, jint)] * model.alloc[i, j] for j, jint in zip(model.M, range(m))))
                                       for i, iint in zip(model.N, range(n))),
                              sense=pyo.maximize)
    opt = pyo.SolverFactory('baron')
    opt.solve(model)
    allocation = []
    for j in model.M:
        already_allocated = False
        for i, iint in zip(model.N, range(n)):
            if pyo.value(model.alloc[i, j]) > .5:
                allocation.append(iint)
                already_allocated = True
                break
        assert already_allocated
    return np.array(allocation)


def maximum_envy(utilities: np.ndarray, allocation: np.ndarray) -> float:
    n, m = utilities.shape
    assert allocation.shape == (m,)
    alloc = np.zeros((m, n), int)
    alloc[np.arange(m), allocation] = 1
    uij = utilities @ alloc
    uii = np.diag(uij)
    envy = uij - uii[:, None]
    return np.amax(envy)


def is_envy_free(utilities: np.ndarray, allocation: np.ndarray) -> bool:
    return maximum_envy(utilities, allocation) <= EPS


def is_ef1(utilities: np.ndarray, allocation: np.ndarray) -> float:
    n, m = utilities.shape
    assert allocation.shape == (m,)
    alloc = np.zeros((m, n), int)
    alloc[np.arange(m), allocation] = 1
    uij = utilities @ alloc
    uii = np.diag(uij)
    envy = uij - uii[:, None]
    most_liked = np.zeros((n, n), float)  # most_liked[i1, i2] = max_{j ￿∈ A_i2} u_i1(j)
    for i1 in range(n):
        for i2 in range(n):
            most_liked[i1, i2] = np.amax(np.multiply(utilities[i1, :], alloc[:, i2]))
    envy_up_to_1 = envy - most_liked

    return np.amax(envy_up_to_1) <= EPS


def is_pareto_optimal(utilities: np.ndarray, allocation: np.ndarray) -> bool:
    n, m = utilities.shape
    model = Model()
    all_vars = []  # index: item, agent
    for j in range(m):
        item_vars = []
        for i in range(n):
            if allocation[j] == i:
                v = model.addVar(vtype=GRB.INTEGER, lb=-1, ub=1)
            else:
                v = model.addVar(vtype=GRB.INTEGER, lb=0, ub=1)
            item_vars.append(v)
        model.addConstr(quicksum(item_vars) == 0)
        all_vars.append(item_vars)
    agent_prefs = []
    for i in range(n):
        i_prefs = quicksum(utilities[i, j] * all_vars[j][i] for j in range(m))
        model.addConstr(i_prefs >= 0.)
        agent_prefs.append(i_prefs)
    model.setObjective(quicksum(agent_prefs), GRB.MAXIMIZE)
    model.optimize()
    return model.objVal < .001


def plot(data, out_path, legend_up: float = 0.):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    plt.xscale("log")
    linestyles = ["solid", "dotted", "dashed", "dashdot", (0, (3, 5, 1, 5)), (0, (5, 1))]
    palette = color_palette("colorblind")
    palette = [palette[2], palette[3], palette[0], palette[4], palette[1]] + palette[5:]
    which_style = 0
    for col in data.columns:
        if col == "m" or col == "iterations":
            continue
        errorbars = np.zeros((2, len(data)), float)
        for i, (successes, iterations) in enumerate(zip(data[col], data["iterations"])):
            left, right = jeffreys_interval(iterations, successes, .95)
            prob = successes / iterations
            errorbars[0, i], errorbars[1, i] = prob - left, right - prob
        label = col.replace("mnw", "MNW").replace("EF ", "envy freeness ").replace("PO", "Pareto optimality")
        ax.errorbar(data["m"], data[col] / data["iterations"], yerr=errorbars, capsize=3.,
                    linestyle=linestyles[which_style % len(linestyles)], color=palette[which_style % len(linestyles)],
                    label=label, ecolor="gray")
        which_style += 1
    ax.autoscale(enable=True)
    plt.grid(True, axis="y")
    ax.set_xlabel("Number of items")
    ax.set_ylabel("Probability")
    plt.legend(loc=(.57, .07 + legend_up), prop={'size': 10})
    plt.tight_layout()
    fig.savefig(out_path)


def experiment_asymmetric(distributions: List[Dist], algorithms: Dict[str, Callable[[np.ndarray], np.ndarray]],
                          iterations: int, m_values: List[float], run_ef1: bool = False) -> pd.DataFrame:
    data = []
    for m in m_values:
        print("m = ", m)
        np.random.seed(m)
        ef_counter = {alg: 0 for alg in algorithms if ((alg == "integral mnw" and m < 500)
                                                       or (alg == "rounded mnw" and m <= 2000) # 2000
                                                       or "mnw" not in alg)}
        if run_ef1:
            ef1_counter = ef_counter.copy()
        else:
            ef1_counter = {}
        po_counter = {alg: 0 for alg in algorithms if alg == "round robin" and m <= 500}
        m_algorithms = {alg: proc for alg, proc in algorithms.items() if (alg in ef_counter or alg in ef1_counter
                                                                          or alg in po_counter)}

        for num_iter in range(iterations):
            if num_iter % 100 == 0:
                print(m, num_iter)
            utilities = np.vstack([dist.sample(m) for dist in distributions])

            allocs = {alg: proc(utilities) for alg, proc in m_algorithms.items()}

            for key, alloc in allocs.items():
                if key in ef_counter and is_envy_free(utilities, alloc):
                    ef_counter[key] += 1
                if key in ef1_counter and is_ef1(utilities, alloc):
                    ef1_counter[key] += 1
                if key in po_counter and is_pareto_optimal(utilities, alloc):
                    po_counter[key] += 1

        datum = {"m": m, "iterations": iterations}
        for key, count in ef_counter.items():
            datum["EF " + key] = count
        for key, count in ef1_counter.items():
            datum["EF1 " + key] = count
        for key, count in po_counter.items():
            datum["PO " + key] = count
        data.append(datum)

    return pd.DataFrame(data)


def experiment_symmetric(dist: Dist, n: int, m: int, iterations: int):
    np.random.seed(m)

    distributions = [dist for _ in range(n)]
    # if all multipliers are one, the multiplier algorithm coincides with welfare-maximizing algorithm:
    multipliers = np.array([1. for _ in distributions])

    ef_count = 0
    for num_iter in range(iterations):
        utilities = np.vstack([dist.sample(m) for dist in distributions])
        alloc = multiplier_allocation(utilities, multipliers)
        if is_envy_free(utilities, alloc):
            ef_count += 1

    return ef_count


def mnw_runtime(distributions: List[Dist], m: int, iterations: int) -> float:
    np.random.seed(m)

    start = time()
    for num_iter in range(iterations):
        print(num_iter)
        utilities = np.vstack([dist.sample(m) for dist in distributions])
        integer_mnw_allocation(utilities)

    return (time() - start) / iterations


def main():
    setParam("OutputFlag", False)

    iterations = 1000

    print("\nStart experiments with triangle-shaped distributions (paper body)")
    num_linear = 10
    base = .1
    distrs = [PiecewiseLinear(i / (num_linear + 1), base) for i in range(1, num_linear + 1)]
    q = 2. - base
    special_distr = distrs[4]
    special_distr_m = 500

    print(f"\nStart computing multipliers {len(distrs)} distributions: {', '.join(str(distr) for distr in distrs)}")
    start = time()
    multipliers = _equalization_multipliers_refine(distrs, .00001, q)
    print(f"Computed multipliers {multipliers} in {time() - start:.0f} seconds.")

    multipliers = np.array(multipliers)
    algorithms = {"multiplier": lambda utils: multiplier_allocation(utils, multipliers),
                  "round robin": round_robin_allocation, "rounded mnw": fractional_mnw_allocation}

    print("\nStart main experiment (Figure 4).")
    ms = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    ms = [m for m in ms if m >= len(distrs)]
    start = time()
    df_main = experiment_asymmetric(distrs, algorithms, iterations,
                                    ms)
    print(f"Completed main experiment in {time() - start:.0f} seconds.")
    df_main.to_csv("fig3.csv", index=False)
    plot(df_main, "fig3.pdf")
    print("Wrote output to fig3.csv and fig3.pdf.")

    print("\nStart experiment with offset values of m (Figure 5).")
    start = time()
    offset_ms = [m + 3 for m in ms if m < 2000]
    df_offset = experiment_asymmetric(distrs, algorithms, iterations, offset_ms)
    print(f"Completed offset experiment in {time() - start:.0f} seconds.")
    df_offset.to_csv("fig5.csv", index=False)
    plot(df_offset, "fig5.pdf")
    print("Wrote output to fig5.csv and fig5.pdf.")

    print(f"\nStarting experiment where {len(distrs)} agents all have the same distribution {special_distr}.")
    start = time()
    ef_count = experiment_symmetric(special_distr, n=len(distrs), m=special_distr_m, iterations=iterations)
    print(f"Finished symmetric experiment in {time() - start:.0f} seconds.")
    lower_conf, upper_conf = jeffreys_interval(iterations, ef_count, .95)
    print(f"Result: At m={special_distr_m}, the welfare-maximizing algorithm is envy free with probability "
          f"~{ef_count / iterations:.2%} (95% confidence interval: [{lower_conf}, {upper_conf}])")

    print("Measuring MNW runtime")
    avg_time = mnw_runtime(distrs, 50, 25)
    print(f"On average, MNW ran in {avg_time} seconds.")

    print("\nStart experiments with Beta distributions (Appendix B.4)")
    distrs = [Beta(.5, .5), Beta(5., 1.), Beta(1., 3.), Beta(2., 2.), Beta(2., 5.)]
    special_distr = distrs[1]
    special_distr_m = 1000
    q = 5.  # not strictly speaking true, in principle computation could diverge

    print(f"\nStart computing multipliers {len(distrs)} distributions: {', '.join(str(distr) for distr in distrs)}")
    start = time()
    multipliers = _equalization_multipliers_refine(distrs, .00001, q)
    print(f"Computed multipliers {multipliers} in {time() - start:.0f} seconds.")

    multipliers = np.array(multipliers)
    algorithms = {"multiplier": lambda utils: multiplier_allocation(utils, multipliers),
                  "round robin": round_robin_allocation, "rounded mnw": fractional_mnw_allocation,
                  "integral mnw": integer_mnw_allocation}

    print("\nStart main experiment (Figure 6).")
    ms = [5, 10, 20, 50, 100, 200]
    ms = [m for m in ms if m >= len(distrs)]
    start = time()
    df_main = experiment_asymmetric(distrs, algorithms, iterations, ms, run_ef1=True)
    print(f"Completed main experiment in {time() - start:.0f} seconds.")
    df_main.to_csv("fig6.csv", index=False)
    df_main = df_main[["m", "iterations", "EF multiplier", "EF round robin", "EF rounded mnw", "EF integral mnw",
                       "EF1 multiplier", "EF1 rounded mnw"]]
    plot(df_main, "fig6.pdf", legend_up=.4)
    print("Wrote output to fig6.csv and fig6.pdf.")

    print(f"\nStarting experiment where {len(distrs)} agents all have the same distribution {special_distr}.")
    start = time()
    ef_count = experiment_symmetric(special_distr, n=len(distrs), m=special_distr_m, iterations=iterations)
    print(f"Finished symmetric experiment in {time() - start:.0f} seconds.")
    lower_conf, upper_conf = jeffreys_interval(iterations, ef_count, .95)
    print(f"Result: At m={special_distr_m}, the welfare-maximizing algorithm is envy free with probability "
          f"~{ef_count / iterations:.2%} (95% confidence interval: [{lower_conf}, {upper_conf}])")


if __name__ == "__main__":
    main()
