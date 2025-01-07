from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.utils.common.result import Ok
from botorch.test_functions.base import BaseTestProblem


def norm(t, T) -> float:
    # normalize to [0, 1]
    x = t / T
    nom = x * 2.0
    return nom - 1.0


def sigmoid_time_transf(t, s, a, T):
    # a: control where to kicks off transformation; lower a, means more range of transformation
    # s: control the speed
    return 1 / (1 + np.exp(-1 * ((norm(t, T) + a) / s)))


class BenchmarkMetric(NoisyFunctionMetric):
    def __init__(
        self,
        name: str,
        param_names: List[str],
        # test_problem: BaseTestProblem,
        noise_sd: float = 0.0,
        index: Optional[int] = 0,
        lower_is_better: Optional[bool] = None,
    ) -> None:
        super().__init__(
            name=name,
            param_names=param_names,
            noise_sd=noise_sd,
            lower_is_better=lower_is_better,
        )
        self.index = index
        # self.test_problem = test_problem

    def clone(self) -> NoisyFunctionMetric:
        return self.__class__(
            name=self._name,
            param_names=self.param_names,
            # test_problem=self.test_problem,
            noise_sd=self.noise_sd,
            index=self.index,
            lower_is_better=self.lower_is_better,
        )

    def fetch_trial_data(
        self, trial: BaseTrial, noisy: bool = True, **kwargs: Any
    ) -> Data:
        noise_sd = self.noise_sd if noisy else 0.0

        mean = [
            trial.run_metadata["Ys"][name][self.index]
            for name, arm in trial.arms_by_name.items()
        ]
        arm_names = [name for name, _ in trial.arms_by_name.items()]

        df = pd.DataFrame(
            {
                "arm_name": arm_names,
                "metric_name": self.name,
                "mean": mean,
                "sem": noise_sd,
                "trial_index": trial.index,
                "n": 10000,
            }
        )
        return Ok(value=Data(df=df))


class UtilBenchmarkMetric(NoisyFunctionMetric):
    def __init__(
        self,
        name: str,
        param_names: List[str],
        test_problem: BaseTestProblem,
        noise_sd: float = 0.0,
        lower_is_better: Optional[bool] = None,
    ) -> None:
        """Obtain true latent util from metric breakwon;"""
        super().__init__(
            name=name,
            param_names=param_names,
            noise_sd=noise_sd,
            lower_is_better=lower_is_better,
        )
        self.test_problem = test_problem

    def clone(self) -> NoisyFunctionMetric:
        return self.__class__(
            name=self._name,
            param_names=self.param_names,
            test_problem=self.test_problem,
            noise_sd=self.noise_sd,
            lower_is_better=self.lower_is_better,
        )

    def fetch_trial_data(
        self, trial: BaseTrial, noisy: bool = True, **kwargs: Any
    ) -> Data:
        noise_sd = self.noise_sd if noisy else 0.0
        mean = [
            # obtain true metrics breakdown and compute util from it
            self.test_problem.compute_util(
                torch.tensor(trial.run_metadata["Ys"][name]).unsqueeze(0)
            )
            for name, arm in trial.arms_by_name.items()
        ]
        arm_names = [name for name, _ in trial.arms_by_name.items()]
        df = pd.DataFrame(
            {
                "arm_name": arm_names,
                "metric_name": self.name,
                "mean": mean,
                "sem": noise_sd,
                "trial_index": trial.index,
                "n": 10000,
            }
        )
        return Ok(value=Data(df=df))


def time_transf_sigmoid_2d(
    x: np.ndarray, running_time: int, converge_time: int, test_problem: BaseTestProblem
) -> float:
    """The function that produces the time-varying effect of the outcome."""
    x_s_dim = 0
    x_a_dim = min(x.shape[-1], 1)

    s = min(
        0.05
        + 0.5
        * (x[x_s_dim] - test_problem._bounds[x_s_dim][0])
        / (test_problem._bounds[x_s_dim][1] - test_problem._bounds[x_s_dim][0]),
        0.5,
    )
    a = 0.8 * (
        (x[x_a_dim] - test_problem._bounds[x_a_dim][0])
        / (test_problem._bounds[x_a_dim][1] - test_problem._bounds[x_a_dim][0])
    )
    return sigmoid_time_transf(running_time, s, a, converge_time)


def aux_time_transf_sigmoid_2d(
    x: np.ndarray,
    running_time: int,
    converge_time: int,
    test_problem: BaseTestProblem,
    x_s_dim: int,
    x_a_dim: int,
) -> float:
    """The function that produces the time-varying effect of the auxillary outcome. It has a same
    formula as time_transf_sigmoid_2d but with larger s and a which will apply weaker time-varying
    effects to these auxillary outcomes (tend to converge faster in the real-world cases).
    """

    s = 0.5 + (x[x_s_dim] - test_problem._bounds[x_s_dim][0]) / (
        test_problem._bounds[x_s_dim][1] - test_problem._bounds[x_s_dim][0]
    )

    a = 2 * (
        (x[x_a_dim] - test_problem._bounds[x_a_dim][0])
        / (test_problem._bounds[x_a_dim][1] - test_problem._bounds[x_a_dim][0])
    )
    return sigmoid_time_transf(running_time, s, a, converge_time)


class TVBenchmarkMetric(NoisyFunctionMetric):
    def __init__(
        self,
        name: str,
        param_names: List[str],
        test_problem: BaseTestProblem,
        converge_time: float,
        time_transf: Callable = time_transf_sigmoid_2d,
        noise_sd: float = 0.0,
        index: Optional[int] = 0,
        lower_is_better: Optional[bool] = None,
    ) -> None:
        super().__init__(
            name=name,
            param_names=param_names,
            noise_sd=noise_sd,
            lower_is_better=lower_is_better,
        )
        self.index = index
        self.test_problem = test_problem
        self.converge_time = converge_time
        self.time_transf = time_transf

    def clone(self) -> NoisyFunctionMetric:
        return self.__class__(
            name=self._name,
            param_names=self.param_names,
            test_problem=self.test_problem,
            converge_time=self.converge_time,
            time_transf=self.time_transf,
            noise_sd=self.noise_sd,
            index=self.index,
            lower_is_better=self.lower_is_better,
        )

    def fetch_trial_data(
        self, trial: BaseTrial, running_time: int, noisy: bool = True, **kwargs: Any
    ) -> Data:
        noise_sd = self.noise_sd if noisy else 0.0
        arm_names = []
        mean = []
        for name, arm in trial.arms_by_name.items():
            arm_names.append(name)
            x = np.array([arm.parameters[p] for p in self.param_names])
            mean.append(
                trial.run_metadata["Ys"][name][self.index]
                * self.time_transf(
                    x, running_time, self.converge_time, self.test_problem
                )
            )
        df = pd.DataFrame(
            {
                "arm_name": arm_names,
                "metric_name": self.name,
                "mean": mean,
                "sem": noise_sd,
                "trial_index": trial.index,
                "n": 10000,
                "start_time": datetime.strptime("2024-01-01", "%Y-%m-%d"),
                "end_time": datetime.strptime("2024-01-01", "%Y-%m-%d")
                + timedelta(days=running_time),
            }
        )
        return Ok(value=Data(df=df))


class TVUtilBenchmarkMetric(NoisyFunctionMetric):
    def __init__(
        self,
        name: str,
        param_names: List[str],
        test_problem: BaseTestProblem,
        converge_time: float,
        time_transf_dict: Dict[str, Callable],
        noise_sd: float = 0.0,
        lower_is_better: Optional[bool] = None,
    ) -> None:
        """
        Args:
            time_transf_dict: A dictionary mapping metric names to functions that apply
            time-varying transformation to the metric
        """
        super().__init__(
            name=name,
            param_names=param_names,
            noise_sd=noise_sd,
            lower_is_better=lower_is_better,
        )
        self.test_problem = test_problem
        self.converge_time = converge_time
        self.time_transf_dict = time_transf_dict

    def clone(self) -> NoisyFunctionMetric:
        return self.__class__(
            name=self._name,
            param_names=self.param_names,
            test_problem=self.test_problem,
            converge_time=self.converge_time,
            time_transf_dict=self.time_transf_dict,
            noise_sd=self.noise_sd,
            lower_is_better=self.lower_is_better,
        )

    def fetch_trial_data(
        self, trial: BaseTrial, running_time: int, noisy: bool = True, **kwargs: Any
    ) -> Data:
        noise_sd = self.noise_sd if noisy else 0.0
        arm_names = []
        mean = []
        for name, arm in trial.arms_by_name.items():
            arm_names.append(name)
            x = np.array([arm.parameters[p] for p in self.param_names])

            tv_outcomes = []
            for idx, metric_name in enumerate(self.test_problem.outcome_list):
                # metric specific time-varying transformation
                time_transf = self.time_transf_dict[metric_name]
                tv_outcomes.append(
                    trial.run_metadata["Ys"][name][idx]
                    * time_transf(
                        x, running_time, self.converge_time, self.test_problem
                    )
                )
            mean.append(
                self.test_problem.compute_util(torch.tensor(tv_outcomes).unsqueeze(0))
            )
        df = pd.DataFrame(
            {
                "arm_name": arm_names,
                "metric_name": self.name,
                "mean": mean,
                "sem": noise_sd,
                "trial_index": trial.index,
                "n": 10000,
                "start_time": datetime.strptime("2024-01-01", "%Y-%m-%d"),
                "end_time": datetime.strptime("2024-01-01", "%Y-%m-%d")
                + timedelta(days=running_time),
            }
        )
        return Ok(value=Data(df=df))
