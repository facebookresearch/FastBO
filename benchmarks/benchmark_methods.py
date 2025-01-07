from copy import deepcopy
from logging import Logger

from typing import List, Optional, Tuple, Union

import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import ComparisonOp, OptimizationConfig
from target_aware_bo import (
    get_and_fit_online_offline_model,
    predict_from_mixed_model,
)
from ax.modelbridge.factory import get_sobol, Models
from ax.modelbridge.registry import Cont_X_trans, ST_MTGP_trans, Y_trans
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transforms.derelativize import Derelativize
from ax.modelbridge.transforms.metrics_as_task import MetricsAsTask
from ax.modelbridge.transforms.stratified_standardize_y import StratifiedStandardizeY
from ax.modelbridge.transforms.task_encode import TaskEncode
from ax.models.torch.botorch import BotorchModel
from ax.models.torch.botorch_defaults import recommend_best_out_of_sample_point
from ax.runners.synthetic import SyntheticRunner
from ax.utils.common.logger import get_logger
from problem_factory import get_synthetic_experiment


logger: Logger = get_logger(__name__)


def run_quickbo(
    synthetic_function_name: str,
    model_name: str,
    noise_sd: float,
    running_time: float,
    num_of_arms: int,
    num_of_long_run_arms: int,
    num_of_short_run_batches: int,
    converge_time: Optional[float] = None,
    seed: Optional[int] = None,
) -> Union[Experiment, Experiment, Data, Data]:
    """
    Args:
        synthetic_function_name: Name of the synthetic function to use.
        model_name: Name of the model to use.
        noise_sd: Standard deviation of the noise added to the synthetic function.
        running_time: # of days to run for each short-run iteration;
        num_of_arms: # of test groups to evaluate/run in parallel
        num_of_short_run_batches: # of short iterations; num_of_batches * running_time = total tuning wall-time
    """
    if converge_time is None:
        converge_time = running_time * num_of_short_run_batches

    test_problem, experiment = get_synthetic_experiment(
        synthetic_function_name=synthetic_function_name,
        noise_sd=noise_sd,
        converge_time=converge_time,
    )
    logger.info(f"run QuickBO with opt-config {experiment.optimization_config}...")

    order_keys = list(experiment.search_space.parameters.keys())
    logger.info(f"optimize parameters {order_keys}...")

    # long-run sobol trial
    m = get_sobol(search_space=experiment.search_space, seed=seed)
    gr = m.gen(n=num_of_long_run_arms)
    trial = experiment.new_batch_trial()
    trial.add_generator_run(generator_run=gr)
    for a in trial.arms:
        # sort the parameter dict based on the search_space.parameters.keys
        # to align with the order of SyntheticTestFunction
        a._parameters = dict(
            sorted(a.parameters.items(), key=lambda x: order_keys.index(x[0]))
        )
    trial.run()
    trial.mark_completed()

    # short-run sobol trial
    num_of_short_run_arms = num_of_arms - num_of_long_run_arms
    gr2 = m.gen(n=num_of_short_run_arms)
    trial = experiment.new_batch_trial()
    trial.add_generator_run(generator_run=gr2)
    for a in trial.arms:
        a._parameters = dict(
            sorted(a.parameters.items(), key=lambda x: order_keys.index(x[0]))
        )
    trial.run()
    trial.mark_completed()

    best_arm_list = []
    for n_iteration in range(1, num_of_short_run_batches):
        # update the long-run trial with longer-term evaluations
        logger.info(
            f"run {n_iteration}th iteration and total running time is {n_iteration*running_time}..."
        )
        data_list = [
            experiment.fetch_trials_data(
                trial_indices=[0],
                metrics=list(experiment.metrics.values()),
                # fetch longer-term evaluations for the long-run trial
                **{"running_time": n_iteration * running_time},
            )
        ]
        data_list.extend(
            [
                experiment.fetch_trials_data(
                    trial_indices=[i],
                    metrics=list(experiment.metrics.values()),
                    # short-term measurements for arms in the short-run trials
                    **{"running_time": running_time},
                )
                for i in list(experiment.trials.keys())[1:]
            ]
        )
        data = Data.from_multiple_data(data_list)

        model, gr = get_model(
            model_name=model_name,
            data=data,
            experiment=experiment,
            n=num_of_short_run_arms,
        )
        # append the recommended arms based on the model prediction
        best_arm_list.append(gr.best_arm_predictions[0])
        # logger.info(f"the best arm from model: {[a.name for a in best_arm_list]}")

        trial = experiment.new_batch_trial()
        trial.add_generator_run(generator_run=gr)
        for a in trial.arms:
            a._parameters = dict(
                sorted(a.parameters.items(), key=lambda x: order_keys.index(x[0]))
            )
        trial.run()
        trial.mark_completed()

    # get all the data and decide the best candidate
    logger.info(
        f"run {n_iteration}th iteration and total running time is {(n_iteration+1)*running_time}..."
    )
    data_list = [
        experiment.fetch_trials_data(
            trial_indices=[0],
            metrics=list(experiment.metrics.values()),
            **{"running_time": (n_iteration + 1) * running_time},
        )
    ]
    data_list.extend(
        [
            experiment.fetch_trials_data(
                trial_indices=[i],
                metrics=list(experiment.metrics.values()),
                **{"running_time": running_time},
            )
            for i in list(experiment.trials.keys())[1:]
        ]
    )
    data = Data.from_multiple_data(data_list)

    # just for fitting model with all the data and selecting the best arm
    model, gr = get_model(
        model_name=model_name,
        data=data,
        experiment=experiment,
        n=1,
    )
    # append the recommended arms based on the model prediction
    best_arm_list.append(gr.best_arm_predictions[0])

    # create an experiment to save the out-of-sample best arm
    experiment_best_arm = Experiment(
        name=experiment.name,
        search_space=experiment.search_space.clone(),
        runner=experiment.runner,
    )
    for a in best_arm_list:
        logger.info(f"the best arm from model: {a.parameters}")
        a._parameters = dict(
            sorted(a.parameters.items(), key=lambda x: order_keys.index(x[0]))
        )
        logger.info(f"the best arm from model after sorting: {a.parameters}")
        t = experiment_best_arm.new_batch_trial(generator_run=GeneratorRun(arms=[a]))
        t.run()
        t.mark_completed()

    best_arm_data = experiment_best_arm.fetch_data(
        metrics=list(experiment.metrics.values()),
        **{"running_time": converge_time},
    )
    return experiment, experiment_best_arm, data, best_arm_data


def run_standard_bo(
    synthetic_function_name: str,
    noise_sd: float,
    running_time: float,
    num_of_arms: int,
    num_of_batches: int,
    converge_time: Optional[float] = None,
    seed: Optional[int] = None,
) -> Union[Experiment, Experiment, Data, Data]:
    """
    Args:
        synthetic_function_name: Name of the synthetic function to use.
        model_name: Name of the model to use.
        noise_sd: Standard deviation of the noise added to the synthetic function.
        running_time: # of days to run for each iteration;
        num_of_arms: # of test groups to evaluate per iteration;
        num_of_batches: # of iterations; num_of_batches * running_time = total tuning wall-time
    """
    if converge_time is None:
        converge_time = running_time * num_of_batches

    test_problem, experiment = get_synthetic_experiment(
        synthetic_function_name=synthetic_function_name,
        noise_sd=noise_sd,
        converge_time=converge_time,
    )
    logger.info(f"run standard BO with opt-config {experiment.optimization_config}...")

    order_keys = list(experiment.search_space.parameters.keys())
    logger.info(f"optimize parameters {order_keys}...")

    m = get_sobol(search_space=experiment.search_space, seed=seed)
    gr = m.gen(n=num_of_arms)
    trial = experiment.new_batch_trial()
    trial.add_generator_run(generator_run=gr)
    for a in trial.arms:
        # sort the parameter dict based on the search_space.parameters.keys
        # to align with the order of SyntheticTestFunction
        a._parameters = dict(
            sorted(a.parameters.items(), key=lambda x: order_keys.index(x[0]))
        )
    trial.run()
    trial.mark_completed()

    best_arm_list = []

    for n_iteration in range(1, num_of_batches):
        logger.info(
            f"run {n_iteration}th iteration and total running time is {n_iteration*running_time}..."
        )

        data_list = [
            experiment.fetch_trials_data(
                trial_indices=[i],
                # all metrics
                metrics=list(experiment.metrics.values()),
                # always observe short-term measurements
                **{"running_time": running_time},
            )
            for i in list(experiment.trials.keys())
        ]
        data = Data.from_multiple_data(data_list)

        # use standard GP
        model = Models.BOTORCH_MODULAR(
            data=data,
            experiment=experiment,
            # model based out-of-sample best arm identification
            out_of_sample_best_point=True,
        )
        gr = model.gen(
            n=num_of_arms,
            optimization_config=experiment.optimization_config,
        )
        # append the recommended arms based on the model prediction
        best_arm_list.append(gr.best_arm_predictions[0])

        trial = experiment.new_batch_trial()
        trial.add_generator_run(generator_run=gr)
        for a in trial.arms:
            a._parameters = dict(
                sorted(a.parameters.items(), key=lambda x: order_keys.index(x[0]))
            )
        trial.run()
        trial.mark_completed()

    # get all the data and decide the best candidate
    logger.info(
        f"run {n_iteration}th iteration and total running time is {(n_iteration+1)*running_time}..."
    )
    data_list = [
        experiment.fetch_trials_data(
            trial_indices=[i],
            # all metrics
            metrics=list(experiment.metrics.values()),
            # always observe short-term measurements
            **{"running_time": running_time},
        )
        for i in list(experiment.trials.keys())
    ]
    data = Data.from_multiple_data(data_list)

    model = Models.BOTORCH_MODULAR(
        data=data,
        experiment=experiment,
        # model based out-of-sample best arm identification
        out_of_sample_best_point=True,
    )
    # unused gr; just for obtaining the out-of-sample best arm
    gr = model.gen(
        n=1,
        optimization_config=experiment.optimization_config,
    )
    # append the recommended arms based on the model prediction
    best_arm_list.append(gr.best_arm_predictions[0])

    # create an experiment to save the out-of-sample best arm
    experiment_best_arm = Experiment(
        name=experiment.name,
        search_space=experiment.search_space.clone(),
        runner=experiment.runner,
    )
    for a in best_arm_list:
        logger.info(f"the best arm from model: {a.parameters}")
        a._parameters = dict(
            sorted(a.parameters.items(), key=lambda x: order_keys.index(x[0]))
        )
        logger.info(f"the best arm from model after sorting: {a.parameters}")
        t = experiment_best_arm.new_batch_trial(generator_run=GeneratorRun(arms=[a]))
        t.run()
        t.mark_completed()
    # fetch true value for these selected out-of-sample best arms
    best_arm_data = experiment_best_arm.fetch_data(
        metrics=list(experiment.metrics.values()),
        **{"running_time": converge_time},
    )
    return experiment, experiment_best_arm, data, best_arm_data


def get_model(
    model_name: str, data: Data, experiment: Experiment, n: int
) -> Tuple[TorchModelBridge, Optional[GeneratorRun]]:
    if model_name == "standard":
        model = Models.BOTORCH_MODULAR(
            data=data,
            experiment=experiment,
            out_of_sample_best_point=True,
        )
        if n > 0:
            gr = model.gen(
                n=n,
                optimization_config=experiment.optimization_config,
            )
        else:
            gr = None
    elif model_name == "icm_default":
        model = Models.BOTORCH_MODULAR(
            data=data,
            experiment=experiment,
            transforms=ST_MTGP_trans,
            out_of_sample_best_point=True,
        )
        if n > 0:
            gr = model.gen(
                n=n,
                optimization_config=experiment.optimization_config,
                fixed_features=ObservationFeatures(parameters={}, trial_index=0),
            )
        else:
            gr = None
    elif model_name == "icm_rank2":
        # create a fake experiment for analysis (combine short-run arms in one trial)
        df = deepcopy(data.df)
        df.loc[df.trial_index > 0, "trial_index"] = 1

        experiment_f = Experiment(
            name="synthetic_experiment",
            search_space=experiment.search_space.clone(),
            runner=SyntheticRunner(),
            optimization_config=experiment.optimization_config.clone(),
        )
        # clone long-run arm and add to trial 0
        experiment_f.new_batch_trial(
            generator_run=GeneratorRun(arms=experiment.trials[0].arms)
        )
        # clone all the short-run arms and add to trial 1
        short_arm_list = []
        for i in range(1, len(experiment.trials)):
            short_arm_list.extend(experiment.trials[i].arms)
        experiment_f.new_batch_trial(generator_run=GeneratorRun(arms=short_arm_list))

        model = Models.BOTORCH_MODULAR(
            data=Data(df=df),
            experiment=experiment_f,
            transforms=ST_MTGP_trans,
            # model based out-of-sample best arm identification
            out_of_sample_best_point=True,
        )
        if n > 0:
            gr = model.gen(
                n=n,
                optimization_config=experiment.optimization_config,
                fixed_features=ObservationFeatures(parameters={}, trial_index=0),
            )
        else:
            gr = None
    elif model_name == "icm_all_proxies":
        df = deepcopy(data.df)
        df.loc[df.trial_index > 0, "trial_index"] = 1
        target_metric_list = list(experiment.optimization_config.metrics)

        aux_metric_list = [m for m in df.metric_name.unique() if "true_" not in m]
        for m in aux_metric_list:
            df.loc[
                (df.metric_name == m) & (df.trial_index > 0), "metric_name"
            ] = f"{m}_proxy"

        experiment_f = Experiment(
            name="synthetic_experiment",
            search_space=experiment.search_space.clone(),
            runner=SyntheticRunner(),
            optimization_config=experiment.optimization_config.clone(),
        )
        # clone long-run arm and add to trial 0
        experiment_f.new_batch_trial(
            generator_run=GeneratorRun(arms=experiment.trials[0].arms)
        )
        # clone all the short-run arms and add to trial 1
        short_arm_list = []
        for i in range(1, len(experiment.trials)):
            short_arm_list.extend(experiment.trials[i].arms)
        experiment_f.new_batch_trial(generator_run=GeneratorRun(arms=short_arm_list))

        mt_proxy_config = {
            "MetricsAsTask": {
                "metric_task_map": {
                    f"{m_p}_proxy": target_metric_list for m_p in aux_metric_list
                }
            }
        }
        logger.info(f"MetricsAsTask config is {mt_proxy_config} for model {model_name}")
        model = Models.BOTORCH_MODULAR(
            data=Data(df=df),
            experiment=experiment_f,
            transforms=Cont_X_trans
            + [
                Derelativize,
                MetricsAsTask,
                StratifiedStandardizeY,
                TaskEncode,
            ],
            transform_configs=mt_proxy_config,
            # model based out-of-sample best arm identification
            out_of_sample_best_point=True,
        )
        if n > 0:
            gr = model.gen(
                n=n,
                optimization_config=experiment.optimization_config,
                fixed_features=ObservationFeatures(parameters={}, trial_index=0),
            )
        else:
            gr = None
    elif model_name in ["ta_gp", "ta_gp_all_proxies"]:
        # modify the metric name in data.df
        target_metric_list = list(experiment.optimization_config.metrics)
        logger.info(f"target metrics are {target_metric_list}")

        # short-term obs as proxy
        if model_name == "ta_gp_all_proxies":
            df = deepcopy(data.df)
            aux_metric_list = [m for m in df.metric_name.unique() if "true_" not in m]
            # use short-term values of all the aux metrics together with target metrics as proxies
            for m in aux_metric_list:
                df.loc[
                    (df.metric_name == m) & (df.trial_index > 0), "metric_name"
                ] = f"{m}_proxy"
            target_metric_config = {
                m: [f"{m}_proxy" for m in aux_metric_list] for m in target_metric_list
            }
        else:
            df = deepcopy(data.df)
            # use short-term values of the target metrics as proxies
            for m in target_metric_list:
                df.loc[
                    (df.metric_name == m) & (df.trial_index > 0), "metric_name"
                ] = f"{m}_proxy"
            target_metric_config = {m: [f"{m}_proxy"] for m in target_metric_list}

        logger.info(
            f"target metric config is {target_metric_config} for model {model_name}"
        )

        model = TorchModelBridge(
            data=Data(df),
            search_space=experiment.search_space,
            experiment=experiment,
            transforms=Cont_X_trans + Y_trans,
            model=BotorchModel(
                model_constructor=get_and_fit_online_offline_model,
                model_predictor=predict_from_mixed_model,
                # model based out-of-sample best arm identification
                best_point_recommender=recommend_best_out_of_sample_point,
                **{
                    "target_metric_config": target_metric_config,
                },
            ),
            torch_dtype=torch.double,
        )
        if n > 0:
            gr = model.gen(
                n=n,
                optimization_config=experiment.optimization_config,
            )
        else:
            gr = None

    return model, gr


def compute_feasibility(
    experiment: Experiment, optimization_config: OptimizationConfig, data: Data
) -> List[str]:
    """Decide whether arms are feasible or not based on the outcome constraints."""
    result_dict = data.df.set_index(["arm_name", "metric_name"]).to_dict()["mean"]
    cons = optimization_config.outcome_constraints
    feas = []
    for arm_name in experiment.arms_by_name.keys():
        g_i = True
        for con in cons:
            if con.op == ComparisonOp.GEQ:
                if result_dict[(arm_name, f"true_{con.metric.name}")] < con.bound:
                    g_i = False
            elif con.op == ComparisonOp.LEQ:
                if result_dict[(arm_name, f"true_{con.metric.name}")] > con.bound:
                    g_i = False

        if g_i:
            feas.append(arm_name)
    return feas
