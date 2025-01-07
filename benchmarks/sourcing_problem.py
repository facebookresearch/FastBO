#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import tempfile
from functools import partial
from typing import Any, Dict, Optional

import numpy as np
import torch
from ax.core.experiment import Experiment
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import FixedParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.runners.botorch_test_problem import BotorchTestProblemRunner

from botorch.test_functions import SyntheticTestFunction
from time_varying_metric import (
    aux_time_transf_sigmoid_2d,
    BenchmarkMetric,
    time_transf_sigmoid_2d,
    TVBenchmarkMetric,
    TVUtilBenchmarkMetric,
    UtilBenchmarkMetric,
)
from torch import Tensor
from torch.distributions import Multinomial


DEFAULT_SOURCING_VAL = 10


def get_sourcing_experiment(
    converge_time: int,
    test_problem_kwargs: Dict[str, Any],
):
    """Create a 6D sourcing experiment for benchmark evaluation.
    The rest 14D parameters are fixed with values being DEFAULT_SOURCING_VAL.

    Args:
        converge_time: The time to which the metrics are expected to converge.
    """
    test_problem = Sourcing()
    active_dim = test_problem.high_quality_indices + test_problem.low_quality_indices
    fixed_val = test_problem_kwargs.get("fixed_val", DEFAULT_SOURCING_VAL)

    # fix the dimension that are not top high-quality or top low-quality sources
    param_list = []
    for i in range(test_problem.dim):
        if i in active_dim:
            param_list.append(
                RangeParameter(
                    name=f"x{i}",
                    parameter_type=ParameterType.INT,
                    lower=test_problem._bounds[i][0],
                    upper=test_problem._bounds[i][1],
                )
            )
        else:
            param_list.append(
                FixedParameter(
                    name=f"x{i}",
                    parameter_type=ParameterType.INT,
                    value=fixed_val,
                )
            )
    search_space = SearchSpace(param_list)
    name = "sourcing"

    obj_outcome_list = [
        "neg_quality_overall",
        "cpu_cost",
    ]
    time_transf_dict = {m: time_transf_sigmoid_2d for m in obj_outcome_list}
    for m in test_problem.outcome_list:
        if m not in obj_outcome_list:
            time_transf_dict[m] = partial(
                aux_time_transf_sigmoid_2d, x_s_dim=1, x_a_dim=2
            )
    outcome_list = test_problem.outcome_list

    objective = Objective(
        metric=TVUtilBenchmarkMetric(
            name="util",
            noise_sd=0.0,
            test_problem=test_problem,
            converge_time=converge_time,
            time_transf_dict=time_transf_dict,
            param_names=list(search_space.parameters.keys()),
        ),
        minimize=True,  # neg quality score
    )
    optimization_config = OptimizationConfig(objective=objective)

    # metric breakdown
    tracking_metrics = [
        UtilBenchmarkMetric(
            name="true_util",
            noise_sd=0,
            test_problem=test_problem,
            param_names=list(search_space.parameters.keys()),
        )
    ] + [
        TVBenchmarkMetric(
            name=metric_name,
            noise_sd=0.0,
            index=test_problem.outcome_list.index(metric_name),
            test_problem=test_problem,
            converge_time=converge_time,
            time_transf=time_transf_dict[metric_name],
            param_names=list(search_space.parameters.keys()),
        )
        for metric_name in outcome_list
    ]

    # true effects
    tracking_metrics += [
        BenchmarkMetric(
            name=f"true_{metric_name}",
            noise_sd=0,
            index=test_problem.outcome_list.index(metric_name),
            param_names=list(search_space.parameters.keys()),
        )
        for metric_name in outcome_list
    ]

    return (
        test_problem,
        Experiment(
            name=f"{name}_experiment",
            search_space=search_space,
            runner=BotorchTestProblemRunner(
                test_problem_class=Sourcing,
                test_problem_kwargs={},
            ),
            optimization_config=optimization_config,
            tracking_metrics=tracking_metrics,
        ),
    )


def sample_items(source_dist, item_dist, fanout_list):
    prob = torch.matmul(source_dist, item_dist)
    s_list = []
    for i, n in enumerate(fanout_list):
        if n > 0:
            s_list.append(Multinomial(n, prob[i, :]).sample().unsqueeze(dim=0))
        else:
            s_list.append(torch.zeros(1, len(prob[i, :])))
    sampled_items = torch.concat(s_list)
    return sampled_items


def compute_ranking_score(sampled_items, item_quality):
    items_to_rank = item_quality.numpy().flatten()[sampled_items.sum(dim=0) > 0]
    final_score_list = np.cumsum(np.sort(items_to_rank)[::-1])
    if len(final_score_list) == 0:
        return 0
    else:
        return final_score_list[-1]


def simulation_one_rep(source_dist, item_dist, fanout_list, item_quality):
    sampled_items = sample_items(source_dist, item_dist, fanout_list)
    # source with high quality topics (> 0.5)
    high_quality_indices = [
        index for index, value in enumerate((source_dist.numpy() > 0.5)[:, 0]) if value
    ]
    low_quality_indices = [
        index for index, value in enumerate((source_dist.numpy() > 0.5)[:, -2]) if value
    ]
    score_dict = {"quality_overall": compute_ranking_score(sampled_items, item_quality)}
    for i in high_quality_indices + low_quality_indices:
        score_dict[f"quality_source_{i}"] = compute_ranking_score(
            sampled_items[[i], :], item_quality
        )
    return score_dict


def simulation(source_dist, item_dist, fanout_list, item_quality, nreps=2000):
    high_quality_indices = [
        index for index, value in enumerate((source_dist.numpy() > 0.5)[:, 0]) if value
    ]
    low_quality_indices = [
        index for index, value in enumerate((source_dist.numpy() > 0.5)[:, -2]) if value
    ]
    outcome_list = ["quality_overall"] + [
        f"quality_source_{i}" for i in high_quality_indices + low_quality_indices
    ]

    result_dict = {k: [] for k in outcome_list}
    for _ in range(nreps):
        score_dict = simulation_one_rep(
            source_dist, item_dist, fanout_list, item_quality
        )
        for k in outcome_list:
            result_dict[k].append(score_dict[k])
    return {k: np.mean(v) for k, v in result_dict.items()}


class Sourcing(SyntheticTestFunction):
    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ):

        self.dim = 25
        self._bounds = [(0, 50) for _ in range(25)]
        self.num_constraints = 7

        with open('sourcing_spec.json', 'r') as file:
            sourcing_spec = json.load(file)

        self.cost_scores = sourcing_spec.get("cost_scores", None)
        self.topic_quality = torch.tensor(sourcing_spec["topic_quality"])
        self.source_dist = torch.tensor(sourcing_spec["source_dist"])
        self.item_dist = torch.tensor(sourcing_spec["item_dist"])
        self.item_quality = torch.matmul(self.topic_quality, self.item_dist)

        # obtain breakdown quality values by sources
        self.high_quality_indices = [
            index
            for index, value in enumerate((self.source_dist.numpy() > 0.5)[:, 0])
            if value
        ]
        self.low_quality_indices = [
            index
            for index, value in enumerate((self.source_dist.numpy() > 0.5)[:, -2])
            if value
        ]
        self.constraint_list = ["cpu_cost"] + [
            f"quality_source_{i}"
            for i in self.high_quality_indices + self.low_quality_indices
        ]
        self.outcome_list = ["neg_quality_overall"] + self.constraint_list

        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        if X.ndim == 1:
            X = X.unsqueeze(0)
        Y = torch.zeros((*X.shape[:-1], 1 + self.num_constraints))
        for sample_idx in range(X.shape[0]):
            # put parameters into 1-D list
            res = simulation(
                self.source_dist,
                self.item_dist,
                [int(x) for x in X[sample_idx, :].numpy()],
                self.item_quality,
                1000,
            )
            res["cpu_cost"] = sum(
                [
                    self.cost_scores[i] * a
                    for i, a in enumerate(X[sample_idx, :].numpy())
                ]
            )
            # -1 * quality (smaller the better)
            Y[sample_idx, :] = torch.tensor(
                [-1 * res["quality_overall"], res["cpu_cost"]]
                + [
                    res[f"quality_source_{i}"]
                    for i in self.high_quality_indices + self.low_quality_indices
                ]
            )
        return Y

    def compute_util(self, Y: Tensor):
        """
        Args:
            Y: tensor of shape (n_points, n_outcomes), where each row is a point
            (containing all the outcomes ordered based on self.outcome_list)
        """
        # util formula - the smaller the better
        util = (
            Y[
                ...,
                self.outcome_list.index("neg_quality_overall"),
            ]
            + 0.6 * Y[..., self.outcome_list.index("cpu_cost")]
            + 20.0 * max(Y[..., self.outcome_list.index("cpu_cost")] - 16.0, 0)
        )
        return util
