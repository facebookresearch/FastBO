#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import re
from typing import Any, Dict, Tuple

from ax.benchmark.benchmark_problem import _get_name

from ax.core.experiment import Experiment
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    ObjectiveThreshold,
    OptimizationConfig,
)
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.types import ComparisonOp
from ax.runners.botorch_test_problem import BotorchTestProblemRunner
from ax.utils.common.typeutils import checked_cast
from botorch.test_functions.base import BaseTestProblem, ConstrainedBaseTestProblem
from botorch.test_functions.multi_objective import (
    BraninCurrin,
    CarSideImpact,
    Penicillin,
)
from botorch.test_functions.synthetic import (
    Ackley,
    Branin,
    ConstrainedGramacy,
    Hartmann,
    Michalewicz,
    PressureVessel,
    Shekel,
    SpeedReducer,
    TensionCompressionString,
    WeldedBeamSO,
)
from contextual_abr_problem import get_contextual_abr_experiment
# from dora_problem import get_dora_experiment
# from ovm_problem import get_ovm_experiment
from sourcing_problem import get_sourcing_experiment
from time_varying_metric import BenchmarkMetric, TVBenchmarkMetric


def get_synthetic_experiment(
    synthetic_function_name, noise_sd, converge_time
) -> Tuple[BaseTestProblem, Experiment]:
    """
    Get an experiment using the specified synthetic function.
    Args:
        synthetic_function_name: Name of the synthetic function to use.
        noise_sd: Standard deviation of the noise added to the synthetic function.
        converge_time: Number of days to take to converge to the true long-term effects.
    Returns:
        An experiment using the specified synthetic function.
    """
    if synthetic_function_name == "sourcing":
        test_problem, experiment = get_sourcing_experiment(
            converge_time=converge_time,
            test_problem_kwargs={},
        )
    elif synthetic_function_name == "dora":
        test_problem, experiment = get_dora_experiment(
            converge_time=converge_time,
            test_problem_kwargs={},
        )
    elif synthetic_function_name == "ovm":
        test_problem, experiment = get_ovm_experiment(
            converge_time=converge_time,
            test_problem_kwargs={},
        )
    elif synthetic_function_name == "contextual_abr_1":
        test_problem, experiment = get_contextual_abr_experiment(
            converge_time=converge_time,
            test_problem_kwargs={},
            # test_problem_kwargs={"num_contexts": 1},
        )
    elif synthetic_function_name == "Shekel":
        test_problem, experiment = from_single_objective(
            test_problem_class=Shekel,
            converge_time=converge_time,
            test_problem_kwargs={"noise_std": noise_sd},
        )
    elif synthetic_function_name == "Branin":
        test_problem, experiment = from_single_objective(
            test_problem_class=Branin,
            converge_time=converge_time,
            test_problem_kwargs={"noise_std": noise_sd},
        )
    elif "Hartmann_" in synthetic_function_name:
        test_problem, experiment = from_single_objective(
            test_problem_class=Hartmann,
            converge_time=converge_time,
            test_problem_kwargs={
                "noise_std": noise_sd,
                "dim": int(re.split("_", synthetic_function_name)[1][0]),
            },
        )
    elif "Ackley_" in synthetic_function_name:
        test_problem, experiment = from_single_objective(
            test_problem_class=Ackley,
            converge_time=converge_time,
            test_problem_kwargs={
                "noise_std": noise_sd,
                "dim": int(re.split("_", synthetic_function_name)[1][0]),
            },
        )
    elif "Michalewicz_" in synthetic_function_name:
        # d = 2, 5, 10
        test_problem, experiment = from_single_objective(
            test_problem_class=Michalewicz,
            converge_time=converge_time,
            test_problem_kwargs={
                "noise_std": noise_sd,
                "dim": int(re.split("_", synthetic_function_name)[1][0]),
            },
        )
    elif synthetic_function_name == "ConstrainedGramacy":
        test_problem, experiment = from_single_objective(
            test_problem_class=ConstrainedGramacy,
            converge_time=converge_time,
            test_problem_kwargs={"noise_std": noise_sd},
        )
    elif synthetic_function_name == "PressureVessel":
        test_problem, experiment = from_single_objective(
            test_problem_class=PressureVessel,
            converge_time=converge_time,
            test_problem_kwargs={"noise_std": noise_sd},
        )
        # modify optimization config
        oc_list = experiment.optimization_config.outcome_constraints
        outcome_constraints = [
            OutcomeConstraint(
                metric=oc_list[0].metric,
                op=ComparisonOp.GEQ,
                bound=3.5,
                relative=False,
            ),
            OutcomeConstraint(
                metric=oc_list[1].metric,
                op=ComparisonOp.GEQ,
                bound=1.5,
                relative=False,
            ),
        ]
        optimization_config = OptimizationConfig(
            objective=experiment.optimization_config.objective,
            outcome_constraints=outcome_constraints,
        )
        experiment.optimization_config = optimization_config
    elif synthetic_function_name == "WeldedBeamSO":
        test_problem, experiment = from_single_objective(
            test_problem_class=WeldedBeamSO,
            converge_time=converge_time,
            test_problem_kwargs={"noise_std": noise_sd},
        )
    elif synthetic_function_name == "TensionCompressionString":
        test_problem, experiment = from_single_objective(
            test_problem_class=TensionCompressionString,
            converge_time=converge_time,
            test_problem_kwargs={"noise_std": noise_sd},
        )
    elif synthetic_function_name == "SpeedReducer":
        test_problem, experiment = from_single_objective(
            test_problem_class=SpeedReducer,
            converge_time=converge_time,
            test_problem_kwargs={"noise_std": noise_sd},
        )
    elif synthetic_function_name == "BraninCurrin":
        test_problem, experiment = from_multi_objective(
            test_problem_class=BraninCurrin,
            converge_time=converge_time,
            test_problem_kwargs={"noise_std": noise_sd},
        )
    elif synthetic_function_name == "Penicillin":
        test_problem, experiment = from_multi_objective(
            test_problem_class=Penicillin,
            converge_time=converge_time,
            test_problem_kwargs={"noise_std": noise_sd},
        )
    elif synthetic_function_name == "CarSideImpact":
        test_problem, experiment = from_multi_objective(
            test_problem_class=CarSideImpact,
            converge_time=converge_time,
            test_problem_kwargs={"noise_std": noise_sd},
        )
    return test_problem, experiment


def from_multi_objective(
    test_problem_class: BaseTestProblem,
    converge_time: int,
    test_problem_kwargs: Dict[str, Any],
) -> Tuple[BaseTestProblem, Experiment]:
    """
    Create a multi-objective experiment with the given test problem from botorch.
    Args:
        test_problem_class: The BoTorch test problem class which will be
            used to define the `search_space`, `optimization_config`, and
            `runner`.
        converge_time: int; the number of days to take to converge to the true long-term effects.
        test_problem_kwargs: Keyword arguments used to instantiate the
            `test_problem_class`.
    """
    test_problem = test_problem_class(**test_problem_kwargs)
    infer_noise = False

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
    dim = test_problem_kwargs.get("dim", None)
    name = _get_name(test_problem, True, dim)

    n_obj = test_problem.num_objectives
    if infer_noise:
        noise_sds = [None] * n_obj
    elif isinstance(test_problem.noise_std, list):
        noise_sds = test_problem.noise_std
    else:
        noise_sds = [checked_cast(float, test_problem.noise_std or 0.0)] * n_obj

    # data observed from the experiments
    metrics = [
        TVBenchmarkMetric(
            name=f"{name}_{i}",
            noise_sd=noise_sd,
            index=i,
            test_problem=test_problem,
            converge_time=converge_time,
            param_names=list(search_space.parameters.keys()),
        )
        for i, noise_sd in enumerate(noise_sds)
    ]

    optimization_config = MultiObjectiveOptimizationConfig(
        objective=MultiObjective(
            objectives=[
                Objective(
                    metric=metric,
                    minimize=True,
                )
                for metric in metrics
            ]
        ),
        objective_thresholds=[
            ObjectiveThreshold(
                metric=metrics[i],
                bound=test_problem.ref_point[i].item(),
                relative=False,
                op=ComparisonOp.LEQ,
            )
            for i in range(test_problem.num_objectives)
        ],
    )
    # true effects
    tracking_metrics = [
        BenchmarkMetric(
            name=f"true_{name}_{i}",
            noise_sd=0,
            index=i,
            # test_problem=test_problem,
            param_names=list(search_space.parameters.keys()),
        )
        for i in range(len(metrics))
    ]
    return (
        test_problem,
        Experiment(
            name=f"{name}_moo_experiment",
            search_space=search_space,
            runner=BotorchTestProblemRunner(
                test_problem_class=test_problem_class,
                test_problem_kwargs=test_problem_kwargs,
            ),
            optimization_config=optimization_config,
            tracking_metrics=tracking_metrics,
        ),
    )


def from_single_objective(
    test_problem_class: BaseTestProblem,
    converge_time: int,
    test_problem_kwargs: Dict[str, Any],
) -> Tuple[BaseTestProblem, Experiment]:
    """
    Create a single-objective experiment with the given test problem from botorch.
    Args:
        test_problem_class: The BoTorch test problem class which will be
            used to define the `search_space`, `optimization_config`, and
            `runner`.
        converge_time: int; the number of days to take to converge to the true long-term effects.
        test_problem_kwargs: Keyword arguments used to instantiate the
            `test_problem_class`.
    """
    test_problem = test_problem_class(**test_problem_kwargs)
    is_constrained = isinstance(test_problem, ConstrainedBaseTestProblem)
    infer_noise = False

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
    dim = test_problem_kwargs.get("dim", None)
    name = _get_name(test_problem, True, dim)

    if isinstance(test_problem.noise_std, list):
        # Convention is to have the first outcome be the objective,
        # and the remaining ones the constraints.
        noise_sd = test_problem.noise_std[0]
    else:
        noise_sd = checked_cast(float, test_problem.noise_std or 0.0)

    # data observed from the experiments
    objective = Objective(
        metric=TVBenchmarkMetric(
            name=f"{name}_0",
            noise_sd=noise_sd,
            index=0,
            test_problem=test_problem,
            converge_time=converge_time,
            param_names=list(search_space.parameters.keys()),
        ),
        minimize=True,
    )

    if is_constrained:
        n_con = test_problem.num_constraints
        if infer_noise:
            constraint_noise_sds = [None] * n_con
        elif test_problem.constraint_noise_std is None:
            constraint_noise_sds = [0.0] * n_con
        elif isinstance(test_problem.constraint_noise_std, list):
            constraint_noise_sds = test_problem.constraint_noise_std[:n_con]
        else:
            constraint_noise_sds = [test_problem.constraint_noise_std] * n_con

        outcome_constraints = [
            OutcomeConstraint(
                metric=TVBenchmarkMetric(
                    name=f"{name}_{i+1}",
                    noise_sd=constraint_noise_sds[i],
                    index=i + 1,
                    test_problem=test_problem,
                    converge_time=converge_time,
                    param_names=list(search_space.parameters.keys()),
                ),
                op=ComparisonOp.GEQ,
                bound=0.0,
                relative=False,
            )
            for i in range(n_con)
        ]

    else:
        outcome_constraints = []

    optimization_config = OptimizationConfig(
        objective=objective,
        outcome_constraints=outcome_constraints,
    )
    # true effects
    tracking_metrics = [
        BenchmarkMetric(
            name=f"true_{name}_{i}",
            noise_sd=0,
            index=i,
            # test_problem=test_problem,
            param_names=list(search_space.parameters.keys()),
        )
        for i in range(len(optimization_config.metrics))
    ]
    return (
        test_problem,
        Experiment(
            name=f"{name}_experiment",
            search_space=search_space,
            runner=BotorchTestProblemRunner(
                test_problem_class=test_problem_class,
                test_problem_kwargs=test_problem_kwargs,
            ),
            optimization_config=optimization_config,
            tracking_metrics=tracking_metrics,
        ),
    )
