#! /usr/bin/env python3

from itertools import product
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.deterministic import PosteriorMeanModel
from botorch.utils.transforms import normalize
from fb_abr_problem import (
    Agent,
    TH_DEFAULT,
    TH_START_DEFAULT,
)
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from park.envs.abr_sim.abr_fb import SessionABRSimFBEnv
from torch import Tensor


class ContextualAgent(Agent):
    def __init__(self, x, **kwargs):
        """Contextual agent Constructor that resets bandwidths, buffer etc for
        different contexts.

        x is an array of parameter values, the structure of which will depend
        on the agent's strategy.
        """
        self.num_encodes = 5
        self.parse_params(x, **kwargs)
        self.reset(context_val=None)

    def parse_params(x, **kwargs):  # noqa
        """
        Should parse x in such a way that set_params_by_context can be run.
        """
        raise NotImplementedError

    def set_params_by_context(self, context_val):
        """
        Should set self.bw, self.bf, self.c, and self.exp_weight
        """
        raise NotImplementedError

    def reset(self, context_val):
        self.prev_bw = []
        self.prev_t = []
        if context_val is not None:
            self.set_params_by_context(context_val)
            self.th = TH_DEFAULT
            self.th_start = TH_START_DEFAULT
            self.th_levels = [
                self.th_start + i * self.th for i in range(self.num_encodes)
            ]
            self.th_levels.append(np.inf)  # avoid empty sequence at loopup


class BucketizedContextualAgent(ContextualAgent):
    def parse_params(self, x, context_buckets):
        """
        There are C context buckets.
        context_buckets is a `n_buckets x d_c`-dim array.
        Containing the bucket midpoints.traces

        x is assumed to be a 4*C length array like
        [bw_1, ..., bw_C, bf_1, ..., bf_C, c_1, ..., c_C, exp_weight_1, ..., exp_weight_C]
        """
        self.context_buckets = context_buckets
        self.bw_arr, self.bf_arr, self.c_arr, self.exp_weight_arr = np.split(x, 4)

    def set_params_by_context(self, context_val):
        # check this
        context_idx = np.argmin(
            np.linalg.norm(context_val[None, :] - self.context_buckets, axis=-1)
        )
        self.bw = self.bw_arr[context_idx]
        self.bf = self.bf_arr[context_idx]
        self.c = self.c_arr[context_idx]
        self.exp_weight = self.exp_weight_arr[context_idx]


class ParkContextualRunner:
    def __init__(self, agent_cls, agent_kwargs, num_traces=500, max_eval=1000):
        # For tracking iterations
        self.fs = []  # History of rewards, stored not as negative reward
        self.n_eval = 0
        self.max_eval = max_eval
        self.agent_cls = agent_cls
        self.agent_kwargs = agent_kwargs
        self.num_traces = num_traces

    def f(self, x):
        """
        Produces an objective value to be _minimized_. (negative reward).

        x is an array of parameters that the Agent should know how to parse.
        """
        if self.n_eval >= self.max_eval:
            raise StopIteration("Evaluation budget exhuasted")

        agent = self.agent_cls(x=x, **self.agent_kwargs)
        context_buckets = self.agent_kwargs["context_buckets"]

        outcome_list = ["reward", "bitrate", "stall_time", "bitrate_change"]
        total_outcome = {m: [] for m in outcome_list}
        contextual_outcome = {
            m: {k: [] for k in range(context_buckets.shape[0])} for m in outcome_list
        }

        env = SessionABRSimFBEnv()
        for i in range(self.num_traces):
            obs, context_val = env.reset(i)
            if len(obs) == 0:
                break
            agent.reset(context_val)
            # decide the context idx
            context_idx = np.argmin(
                np.linalg.norm(context_val[None, :] - context_buckets, axis=-1)
            )
            done = False
            outcome = {m: 0 for m in outcome_list}
            while not done:
                act = agent.get_action(obs)
                obs, reward, done, info = env.step(act)
                outcome["reward"] += reward
                # include tracking metrics info
                for m in outcome_list[1:]:
                    outcome[m] += info[m]
            for m in outcome_list:
                total_outcome[m].append(outcome[m])
                contextual_outcome[m][context_idx].append(outcome[m])

        overall_outcome = {k: np.mean(v) for k, v in total_outcome.items()}
        self.fs.append(overall_outcome["reward"])
        overall_outcome["neg_reward"] = -1 * overall_outcome["reward"]  # flip the sign

        contextual_outcome_dict = {}
        for m, context_data_dict in contextual_outcome.items():
            contextual_outcome_dict[m] = {
                k: (np.sum(v) / len(total_outcome[m]))
                for k, v in context_data_dict.items()
            }
        contextual_outcome_dict["neg_reward"] = {
            k: (-1 * np.sum(v) / len(total_outcome[m]))
            for k, v in contextual_outcome["reward"].items()
        }

        self.n_eval += 1
        return overall_outcome, contextual_outcome_dict


def get_parameter_specs(num_buckets: int) -> List[Dict[str, Any]]:
    parameter_specs = []
    for c in range(num_buckets):
        parameter_specs.extend(
            [
                {
                    "name": f"bw_c{c}",
                    "type": "range",
                    "bounds": [0.0, 1.0],
                    "value_type": "float",
                    "log_scale": False,
                },
                {
                    "name": f"bf_c{c}",
                    "type": "range",
                    "bounds": [0.0, 3.0],
                    "value_type": "float",
                    "log_scale": False,
                },
                {
                    "name": f"c_c{c}",
                    "type": "range",
                    "bounds": [0.0, 1.0],
                    "value_type": "float",
                    "log_scale": False,
                },
                {
                    "name": f"exp_weight_c{c}",
                    "type": "range",
                    "bounds": [0.0001, 0.25],
                    "value_type": "float",
                    "log_scale": False,
                },
            ]
        )
    return parameter_specs


def get_bucket_decomposition(num_buckets: int) -> Dict[str, List[str]]:
    return {
        f"c{c}": [f"bw_c{c}", f"bf_c{c}", f"c_c{c}", f"exp_weight_c{c}"]
        for c in range(num_buckets)
    }


def get_contextual_buckets_and_bounds(trace_bws, avg_latencies, num_buckets):
    # get quantile for each feature
    num_buckets_per_feat = int(num_buckets**0.5)
    bw_quantiles = np.quantile(
        list(trace_bws.values()), np.linspace(0, 1, num_buckets_per_feat + 2)
    )
    latency_quantiles = np.quantile(
        list(avg_latencies.values()), np.linspace(0, 1, num_buckets_per_feat + 2)
    )
    context_buckets = np.array(
        list(product(bw_quantiles[1:-1], latency_quantiles[1:-1]))
    )
    # get the min and max values for each contextual feature
    context_bounds = np.array(
        [
            [bw_quantiles[0], latency_quantiles[0]],
            [bw_quantiles[-1], latency_quantiles[-1]],
        ]
    )
    return context_buckets, context_bounds
