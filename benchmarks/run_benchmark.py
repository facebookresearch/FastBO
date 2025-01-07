from logging import Logger
from typing import Any, Dict, List, Optional

from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.runners.synthetic import SyntheticRunner
from ax.storage.json_store.encoder import object_to_json
from ax.utils.common.logger import get_logger
from benchmark_methods import run_quickbo, run_standard_bo,

logger: Logger = get_logger(__name__)


def run_benchmark(
    synthetic_function_name: str,
    strategy_name: str,
    strategies_args: Dict[str, Any],
    noise_sd: float,
    running_time: float,
    num_of_arms: int,
    num_of_batches: int,
    converge_time: Optional[float] = None,
    seed: Optional[int] = None,
) -> Dict:
    design_name = strategies_args.get("design_name", None)
    if design_name == "quickbo":
        model_name = strategies_args.get("model_name", "icm_default")
        num_of_long_run_arms = strategies_args.get(
            "num_of_long_run_arms", int(num_of_arms * 0.5)
        )
        logger.info(
            f"run QuickBO design with {num_of_long_run_arms} using {model_name} model..."
        )
        experiment, experiment_best_arm, data, best_arm_data = run_quickbo(
            synthetic_function_name=synthetic_function_name,
            model_name=model_name,
            noise_sd=noise_sd,
            running_time=running_time,
            num_of_arms=num_of_arms,
            num_of_long_run_arms=num_of_long_run_arms,
            num_of_short_run_batches=num_of_batches,
            converge_time=converge_time,
            seed=seed,
        )
    elif design_name == "standard":
        logger.info("run standard BO design ...")
        experiment, experiment_best_arm, data, best_arm_data = run_standard_bo(
            synthetic_function_name=synthetic_function_name,
            noise_sd=noise_sd,
            running_time=running_time,
            num_of_arms=num_of_arms,
            num_of_batches=num_of_batches,
            converge_time=converge_time,
            seed=seed,
        )
    else:
        raise ValueError(f"{design_name} is not supported.")

    # dup an experiment for storage
    experiment_f = Experiment(
        name=experiment.name,
        search_space=experiment.search_space.clone(),
        runner=SyntheticRunner(),
    )
    for t in experiment.trials.values():
        experiment_f.new_batch_trial(generator_run=GeneratorRun(arms=t.arms))
    experiment_f.attach_data(data)

    # create an experiment for storing best selected arm
    experiment_best_arm_f = Experiment(
        name=experiment_best_arm.name,
        search_space=experiment_best_arm.search_space.clone(),
        runner=SyntheticRunner(),
    )
    for t in experiment_best_arm.trials.values():
        experiment_best_arm_f.new_batch_trial(generator_run=GeneratorRun(arms=t.arms))
    experiment_best_arm_f.attach_data(best_arm_data)

    return {
        "experiment": object_to_json(experiment_f),
        "experiment_best_arm": object_to_json(experiment_best_arm_f),
    }


def run_benchmark_reps(
    synthetic_function_name: str,
    strategies: Dict[str, Dict],
    noise_sd: float,
    running_time: float,
    num_of_batches: int,
    num_of_arms: int,
    reps: int = 1,
    converge_time: Optional[float] = None,
) -> Dict[str, List[Dict]]:
    res_all = {s: [] for s in strategies}
    for irep in range(reps):
        # fix seed for each strategy
        for strategy, strategies_args in strategies.items():
            res_all[strategy].append(
                run_benchmark(
                    synthetic_function_name=synthetic_function_name,
                    strategy_name=strategy,
                    strategies_args=strategies_args,
                    noise_sd=noise_sd,
                    num_of_arms=num_of_arms,
                    running_time=running_time,
                    num_of_batches=num_of_batches,
                    converge_time=converge_time,
                    seed=irep * 5,
                )
            )
    return res_all

if __name__ == "__main__":
    if len(sys.argv) != 2:
        current_filename = os.path.basename(__file__)
        print(f"Usage: python {current_filename} <json_file>")
        sys.exit(1)

    json_file = sys.argv[1]

    with open(json_file, "r") as file:
        config = json.load(file)

    output_data = run_benchmark_reps(**config)

    # Specify the output JSON file name
    output_json_file = "benchmark_reps_result.json"
    # Write the output data to a JSON file
    with open(output_json_file, "w") as file:
        json.dump(output_data, file, indent=4)
    print(f"Output has been written to {output_json_file}")
