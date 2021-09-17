from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isinf, inf, isnan
from time import time
from typing import Iterable, List, Tuple, Optional, Dict, Callable

import cvxpy as cp
from gurobipy import Model, quicksum, GRB, setParam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


@dataclass
class BinaryProgression:
    point_to_try: float
    recurse_left: Tuple[float, float]
    recurse_right: Tuple[float, float]


def _binary_progression(left, right, first_multiplier):
    """Defines which step to take next. Starts with an doubling-argument search to find a finite upper bound,
    then uses the bisection method."""
    if isinf(right):
        if left == 0.:
            return BinaryProgression(first_multiplier, (left, first_multiplier), (first_multiplier, right))
        else:
            # if still no value with probability greater than target found, try double the previous point
            return BinaryProgression(2 * left, (left, 2 * left), (2 * left, right))
    else:
        # bisection method
        mid = (left + right) / 2
        return BinaryProgression(mid, (left, mid), (mid, right))


def _equalization_multipliers(distributions: List[Dist], start_multiplier: float, fixed_distributions: List[Dist],
                              fixed_multipliers: List[float], binary_search_iterations: int) -> List[float]:
    assert binary_search_iterations > 0

    if len(distributions) == 0:
        return []
    if len(fixed_distributions) == 0:
        dist = distributions[0]
        recursive_mults = _equalization_multipliers(distributions[1:], start_multiplier, [dist], [1.],
                                                    binary_search_iterations)
        return [1.] + recursive_mults
    else:
        dist = distributions[0]
        n = len(distributions) + len(fixed_distributions)

        left = 0.
        right = inf

        for _ in range(binary_search_iterations):
            prog = _binary_progression(left, right, start_multiplier)
            ptt = prog.point_to_try

            recursive_mults = _equalization_multipliers(distributions[1:], start_multiplier, fixed_distributions+[dist],
                                                        fixed_multipliers + [ptt], binary_search_iterations)

            def local_eval(x: float) -> float:
                prod = dist.pdf(x)
                for dist2, mult in zip(fixed_distributions, fixed_multipliers):
                    prod *= dist2.cdf(ptt / mult * x)
                for dist2, mult in zip(distributions[1:], recursive_mults):
                    prod *= dist2.cdf(ptt / mult * x)
                return prod

            prob = integrate.quad(local_eval, dist.min, dist.max)[0]
            if prob < 1/n:
                left, right = prog.recurse_right
            else:
                left, right = prog.recurse_left

        return [ptt] + recursive_mults


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


def plot(data, out_path):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    plt.xscale("log")
    linestyles = ["solid", "dotted", "dashed", "dashdot"]
    palette = color_palette("colorblind")
    palette = [palette[2], palette[3], palette[0], palette[4]]
    which_style = 0
    for col in data.columns:
        if col == "m" or col == "iterations":
            continue
        errorbars = np.zeros((2, len(data)), float)
        for i, (successes, iterations) in enumerate(zip(data[col], data["iterations"])):
            left, right = jeffreys_interval(iterations, successes, .95)
            prob = successes / iterations
            errorbars[0, i], errorbars[1, i] = prob - left, right - prob
        label = col.replace("EF", "envy freeness").replace("PO", "Pareto optimality").replace("mnw", "MNW")
        ax.errorbar(data["m"], data[col] / data["iterations"], yerr=errorbars, capsize=3.,
                    linestyle=linestyles[which_style % len(linestyles)], color=palette[which_style % len(linestyles)],
                    label=label, ecolor="gray")
        which_style += 1
    ax.autoscale(enable=True)
    plt.grid(True, axis="y")
    ax.set_xlabel("Number of items")
    ax.set_ylabel("Probability")
    plt.legend(loc=(.52, .07), prop={'size': 10})
    fig.savefig(out_path)


def experiment_asymmetric(distributions: List[Dist], algorithms: Dict[str, Callable[[np.ndarray], np.ndarray]],
                          iterations: int, m_values: List[float]) -> pd.DataFrame:
    data = []
    for m in m_values:
        print("m = ", m)
        np.random.seed(m)
        ef_counter = {alg: 0 for alg in algorithms if alg != "mnw" or m <= 2000}
        po_counter = {alg: 0 for alg in algorithms if alg == "round robin" and m <= 2000}

        for num_iter in range(iterations):
            utilities = np.vstack([dist.sample(m) for dist in distributions])

            allocs = {alg: proc(utilities) for alg, proc in algorithms.items()}

            for key, alloc in allocs.items():
                if key in ef_counter and is_envy_free(utilities, alloc):
                    ef_counter[key] += 1
                if key in po_counter and is_pareto_optimal(utilities, alloc):
                    po_counter[key] += 1

        datum = {"m": m, "iterations": iterations}
        for key, count in ef_counter.items():
            datum["EF " + key] = count
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


def main():
    setParam("OutputFlag", False)

    wikipedia_distrs = [Beta(.5, .5), Beta(5., 1.), Beta(1., 3.), Beta(2., 2.), Beta(2., 5.)]
    iterations = 1000

    print("\nStart computing multipliers for the five beta distributions.")
    start = time()
    multipliers = _equalization_multipliers(wikipedia_distrs, 2., [], [], 20)
    print(f"Computed multipliers {multipliers} in {time() - start:.0f} seconds.")
    print("Reference output: [1.0, 0.8339729309082031, 2.0838546752929688, 1.2098731994628906, 2.0375137329101562]")

    multipliers = np.array(multipliers)
    algorithms = {"multiplier": lambda utils: multiplier_allocation(utils, multipliers),
                  "round robin": round_robin_allocation, "mnw": fractional_mnw_allocation}

    print("\nStart main experiment (Figure 4).")
    start = time()
    df_main = experiment_asymmetric(wikipedia_distrs, algorithms, iterations,
                                    [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
    print(f"Completed main experiment in {time() - start:.0f} seconds.")
    df_main.to_csv("fig4.csv", index=False)
    plot(df_main, "fig4.pdf")
    print("Wrote output to fig4.csv and fig4.pdf.")

    print("\nStart experiment with offset values of m (Figure 5).")
    start = time()
    df_offset = experiment_asymmetric(wikipedia_distrs, algorithms, iterations,
                                      [5+3, 10+3, 20+3, 50+3, 100+3, 200+3, 500+3, 1000+3])
    print(f"Completed offset experiment in {time() - start:.0f} seconds.")
    df_main.to_csv("fig5.csv", index=False)
    plot(df_main, "fig5.pdf")
    print("Wrote output to fig5.csv and fig5.pdf.")

    print("\nStarting experiment where five agents all have distribution Beta(5, 1).")
    ef_count = experiment_symmetric(Beta(5., 1.), n=5, m=1000, iterations=iterations)
    print(f"Finished symmetric experiment in {time() - start:.0f} seconds.")
    lower_conf, upper_conf = jeffreys_interval(iterations, ef_count, .95)
    print(f"Result: At m=1000, the welfare-maximizing algorithm is envy free with probability "
          f"~{ef_count / iterations:.2%} (95% confidence interval: [{lower_conf}, {upper_conf}])")


if __name__ == "__main__":
    main()