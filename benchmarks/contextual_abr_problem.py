#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

import numpy as np
import torch

from ax.core.experiment import Experiment
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.runners.botorch_test_problem import BotorchTestProblemRunner
from botorch.test_functions import SyntheticTestFunction
from session_level_abr_problem import (
    BucketizedContextualAgent,
    get_contextual_buckets_and_bounds,
    ParkContextualRunner,
)
from time_varying_metric import (
    BenchmarkMetric,
    time_transf_sigmoid_2d,
    TVBenchmarkMetric,
    TVBenchmarkMetric,
    TVUtilBenchmarkMetric,
    UtilBenchmarkMetric,
)
from park.envs.abr_sim.fb_trace_loader import (
    get_avg_latency_by_trace,
    load_individual_traces,
)
from torch import Tensor


def get_contextual_abr_experiment(
    converge_time: int,
    test_problem_kwargs: Dict[str, Any],
):
    """Create a ABR controller optimization experiment for benchmark evaluation.
    In this case, we optimize global parameters and can obtain context-level outceoms as
    aux metrics (num of contexts = 4).

    Args:
        converge_time: The time to which the metrics are expected to converge.
    """
    test_problem = ContextualSessionABR(**test_problem_kwargs)

    search_space = SearchSpace(
        parameters=[
            RangeParameter(
                name=f"x{i}",
                parameter_type=ParameterType.FLOAT,
                lower=test_problem._bounds[i][0],
                upper=test_problem._bounds[i][1],
            )
            for i in range(test_problem.dim)
        ]
    )
    num_contexts = test_problem_kwargs.get("num_contexts")
    name = f"contextual_abr_{num_contexts}"

    objective = Objective(
        metric=TVUtilBenchmarkMetric(
            name="util",
            noise_sd=0.0,
            test_problem=test_problem,
            converge_time=converge_time,
            # in ABR case, I apply same transformation to all metrics because aux metrics are
            # the context-level breakdown
            time_transf_dict={
                m: time_transf_sigmoid_2d for m in test_problem.outcome_list
            },
            param_names=list(search_space.parameters.keys()),
        ),
        minimize=True,
    )
    optimization_config = OptimizationConfig(objective=objective)

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
            index=idx,
            test_problem=test_problem,
            converge_time=converge_time,
            param_names=list(search_space.parameters.keys()),
        )
        for idx, metric_name in enumerate(test_problem.outcome_list)
    ]

    # true effects
    tracking_metrics += [
        BenchmarkMetric(
            name=f"true_{metric_name}",
            noise_sd=0,
            index=idx,
            param_names=list(search_space.parameters.keys()),
        )
        for idx, metric_name in enumerate(test_problem.outcome_list)
    ]

    return (
        test_problem,
        Experiment(
            name=f"{name}_experiment",
            search_space=search_space,
            runner=BotorchTestProblemRunner(
                test_problem_class=ContextualSessionABR,
                test_problem_kwargs=test_problem_kwargs,
            ),
            optimization_config=optimization_config,
            tracking_metrics=tracking_metrics,
        ),
    )


class ContextualSessionABR(SyntheticTestFunction):
    def __init__(
        self,
        # num_contexts: int = 1,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ):
        # make sure that param_names list is in the same order with original_bounds
        # self.param_names = [
        #     f"{p}_c{c}"
        #     for p in ["bw", "bf", "c", "exp_weight"]
        #     for c in range(num_contexts)
        # ]
        # "bw", "bf", "c", "exp_weight"
        # set in the same order as BucketizedContextualAgent
        # x is assumed to be a 4*C length array like
        # [bw_1, ..., bw_C, bf_1, ..., bf_C, c_1, ..., c_C, exp_weight_1, ..., exp_weight_C]
        num_contexts = 1
        _bounds = torch.tensor(
            [
                [0.0 for _ in range(num_contexts)]
                + [0.0 for _ in range(num_contexts)]
                + [0.0 for _ in range(num_contexts)]
                + [0.0001 for _ in range(num_contexts)],
                [1.0 for _ in range(num_contexts)]
                + [3.0 for _ in range(num_contexts)]
                + [1.0 for _ in range(num_contexts)]
                + [0.25 for _ in range(num_contexts)],
            ]
        )
        self._bounds = [
            (float(_bounds[0, i]), float(_bounds[1, i]))
            for i in range(_bounds.shape[-1])
        ]
        self.dim = 4 * num_contexts
        self.num_contexts = num_contexts
        super().__init__(noise_std=noise_std, negate=negate)

        self.num_contexts_eval = 4

        _, trace_bws = load_individual_traces()
        trace_ids_to_avg_latency = get_avg_latency_by_trace()
        context_buckets, _ = get_contextual_buckets_and_bounds(
            trace_bws, trace_ids_to_avg_latency, self.num_contexts_eval
        )

        self.runner = ParkContextualRunner(
            agent_cls=BucketizedContextualAgent,
            agent_kwargs={"context_buckets": context_buckets},
            max_eval=1000,
        )
        self.overall_outcome_list = [
            "neg_reward",
            "bitrate",
            "stall_time",
            "bitrate_change",
        ]
        self.contextual_outcome_list = [
            f"{m}:c{c}"
            for c in range(self.num_contexts_eval)
            for m in ["bitrate", "stall_time", "bitrate_change"]
        ]
        self.outcome_list = self.overall_outcome_list + self.contextual_outcome_list

    def evaluate_true(self, X: Tensor) -> Tensor:
        # the first self.dim inputs are active parameters
        if X.ndim == 1:
            X = X.unsqueeze(0)
        Y = torch.zeros((*X.shape[:-1], len(self.outcome_list)))
        for sample_idx in range(X.shape[0]):
            # put parameters into 1-D array
            global_X = X[sample_idx, :].numpy()
            overall_outcome, contextual_outcome = self.runner.f(
                np.array([x for x in global_X for _ in range(self.num_contexts_eval)])
            )
            # order by self.outcome_list
            Y[sample_idx, :] = torch.tensor(
                [overall_outcome[k] for k in self.overall_outcome_list]
                + [
                    contextual_outcome[m][c]
                    for c in range(self.num_contexts_eval)
                    for m in ["bitrate", "stall_time", "bitrate_change"]
                ]
            )
        return Y.to(torch.double)

    def compute_util(self, Y: Tensor):
        """
        Args:
            Y: tensor of shape (n_points, n_outcomes), where each row is a point
            (containing all the outcomes ordered based on self.outcome_list)
        """
        # util formula: minimize neg_reward, s.t. stall_time <= 40 and bitrate_change <= 15
        # apply 20 as penalty for violating constraints
        util = (
            Y[
                ...,
                self.outcome_list.index("neg_reward"),
            ]
            + 20.0 * max(Y[..., self.outcome_list.index("stall_time")] - 40.0, 0)
            + 20.0 * max(Y[..., self.outcome_list.index("bitrate_change")] - 15, 0)
        )
        return util
