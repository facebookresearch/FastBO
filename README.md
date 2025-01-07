# FastBO
Code for paper Experimenting, Fast and Slow Bayesian Optimization of Long-term Outcomes with Online Experiments

## Installation
To install the code clone the repo and install the dependencies as

    cd FastBO
    python3 -m pip install -r requirements.txt


### Running Benchmarks
The `benchmarks/` directory contains code for running the numerical experiments described in the paper. The benchmark problems are defined in `problem_facotory.py` and the file `benchmark_spec.json` contains the input config of running benchmarks. To run these experiments

```bash
python3 run_benchmark benchmark_spec.json
```
