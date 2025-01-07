# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from ax.models.torch.botorch import (
    BotorchModel,
    TAcqfConstructor,
    TBestPointRecommender,
    TModelConstructor,
    TModelPredictor,
    TOptimizer,
)
from ax.models.torch.botorch_defaults import (
    _get_model,
    get_and_fit_model,
    get_NEI,
    recommend_best_observed_point,
    scipy_optimizer,
)
from ax.models.torch.botorch_moo import MultiObjectiveBotorchModel
from ax.models.torch.botorch_moo_defaults import get_NEHVI
from botorch.fit import fit_fully_bayesian_model_nuts, fit_gpytorch_model
from botorch.models.fully_bayesian import MCMC_DIM, SaasFullyBayesianSingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import ModelList
from botorch.posteriors.fully_bayesian import FullyBayesianPosterior
from botorch.posteriors.posterior_list import PosteriorList
from map_saas import get_fitted_map_saas_ensemble
from target_aware_gp import TargetAwareEnsembleGP
from gpytorch.mlls import ExactMarginalLogLikelihood, LeaveOneOutPseudoLikelihood
from torch import Tensor

WARMUP_STEPS = 512
NUM_SAMPLES = 256
THINNING = 16


def get_and_fit_online_offline_model(
    Xs: List[Tensor],
    Ys: List[Tensor],
    Yvars: List[Tensor],
    task_features: List[int],
    fidelity_features: List[int],
    metric_names: List[str],
    state_dict: Optional[Dict[str, Tensor]] = None,
    refit_model: bool = True,
    use_input_warping: bool = False,
    use_loocv_pseudo_likelihood: bool = False,
    prior: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Union[GPyTorchModel, ModelList]:
    r"""Instantiates and fits a botorch GPyTorchModel using the given data. When
    `target_metric_config` is given as a kwarg, we will fit a target-aware GP for
    specified target metrics.

    Args:
        Xs: List of X data, one tensor per outcome.
        Ys: List of Y data, one tensor per outcome.
        Yvars: List of observed variance of Ys.
        task_features: List of columns of X that are tasks.
        fidelity_features: List of columns of X that are fidelity parameters.
        metric_names: Names of each outcome Y in Ys.
        state_dict: If provided, will set model parameters to this state
            dictionary. Otherwise, will fit the model.
        refit_model: Flag for refitting model.
        prior: Optional[Dict]. A dictionary that contains the specification of
            GP model prior.
        kwargs: Additional arguments to pass to fit GP model.
            - target_metric_config: A Dict[str, List] that keys are the target metric
            names and values are lists of auxiliary metric names, e.g.
            {"target_metric": ["aux_m1", "aux_m2"]}.
            - target_aware_model_kwargs: A Dict that contains additional arguments
            passed to fit the target aware GP model. Currently, the following
            arguments are supported: {"ensemble_weight_prior": HalfCauchyPrior(0.1)}.

    Returns:
        A fitted GPyTorchModel.
    """
    target_metric_config = kwargs.pop("target_metric_config", {})
    target_aware_model_kwargs = kwargs.pop("target_aware_model_kwargs", {})

    # if no target metric is specified, use standard botorch model_constructor
    if len(target_metric_config) == 0:
        return get_and_fit_model(
            Xs=Xs,
            Ys=Ys,
            Yvars=Yvars,
            task_features=task_features,
            fidelity_features=fidelity_features,
            metric_names=metric_names,
            state_dict=state_dict,
            refit_model=refit_model,
            use_input_warping=use_input_warping,
            use_loocv_pseudo_likelihood=use_loocv_pseudo_likelihood,
            prior=prior,
            **kwargs,
        )

    if len(fidelity_features) > 0:
        raise NotImplementedError(
            "Currently do not support Target-aware GP with fidelty features!"
        )

    if len(task_features) == 1:
        task_feature = task_features[0]
    else:
        task_feature = None

    target_metric_list = list(target_metric_config.keys())

    model_dict = {}  # a dict of model only using single metric for modeling
    for i, X in enumerate(Xs):
        if metric_names[i] not in target_metric_list:
            # get and fit the model for non-target metrics
            model_dict[metric_names[i]] = _get_fitted_gp_model(
                X=X,
                Y=Ys[i],
                Yvar=Yvars[i],
                task_feature=task_feature,
                use_input_warping=use_input_warping,
                use_loocv_pseudo_likelihood=use_loocv_pseudo_likelihood,
                prior=prior,
                state_dict=state_dict,
                **kwargs,
            )

    model_list = []
    for i, X in enumerate(Xs):
        if metric_names[i] not in target_metric_list:
            model_list.append(model_dict[metric_names[i]])
        else:
            # TODO: update state_dict
            aux_metric_list = target_metric_config[metric_names[i]]
            # check the auxillary metrics are fitted as part of model_dict
            assert all(
                m in metric_names for m in aux_metric_list
            ), "Auxiliary metrics are not part of the training data."
            m = _get_fitted_target_aware_gp(
                train_X=X,
                train_Y=Ys[i],
                train_Yvar=Yvars[i],
                base_model_dict={
                    metric_name: model_dict[metric_name]
                    for metric_name in aux_metric_list
                },
                state_dict=state_dict,
                **target_aware_model_kwargs,
            )
            model_list.append(m)
    model = ModelList(*model_list)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.to(Xs[0])
    return model


def _get_fitted_target_aware_gp(
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor,
    base_model_dict: Dict[str, GPyTorchModel],
    state_dict: Optional[Dict[str, Tensor]] = None,
    **kwargs: Any,
) -> TargetAwareEnsembleGP:
    r"""Construct and fit target-aware GP.

    Args:
        train_X: A `n x d` tensor of training inputs of target task.
        train_Y: A `n x 1` tensor of training outputs of target task.
        train_Yvar : A `n x 1` tensor of observed variance of target task.
        base_model_dict: A dictionary of GP models fitted on auxillary data sources.
        state_dict: The state dict of the target-aware GP model.
    """
    ensemble_gp = TargetAwareEnsembleGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        base_model_dict=deepcopy(base_model_dict),
        ensemble_weight_prior=kwargs.get("ensemble_weight_prior", None),
    )
    mll = LeaveOneOutPseudoLikelihood(ensemble_gp.likelihood, ensemble_gp)
    if state_dict is None:
        fit_gpytorch_model(mll)
    return ensemble_gp


def _get_fitted_gp_model(
    X: Tensor,
    Y: Tensor,
    Yvar: Tensor,
    task_feature: Optional[int] = None,
    use_input_warping: bool = False,
    use_loocv_pseudo_likelihood: bool = False,
    prior: Optional[Dict[str, Any]] = None,
    state_dict: Optional[Dict[str, Tensor]] = None,
    **kwargs: Any,
) -> GPyTorchModel:
    r"""Get a fitted GP for a single metric depending on the input data and
    specified model type.

    Args:
        X: A `n x d` tensor of input features.
        Y: A `n x m` tensor of input observations.
        Yvar: A `n x m` tensor of input variances (NaN if unobserved).
        task_feature: The index of the column pertaining to the task feature
            (if present).
        prior: Optional[Dict]. A dictionary that contains the specification of
            GP model prior. Currently, the keys include:
            - covar_module_prior: prior on covariance matrix e.g.
                {"lengthscale_prior": GammaPrior(3.0, 6.0)}.
            - type: type of prior on task covariance matrix e.g.`LKJCovariancePrior`.
            - sd_prior: A scalar prior over nonnegative numbers, which is used for the
                default LKJCovariancePrior task_covar_prior.
            - eta: The eta parameter on the default LKJ task_covar_prior.
        state_dict: If provided, will set model parameters to this state
            dictionary. Otherwise, will fit the model.
        kwargs: Additional arguments to pass to fit GP model.
            - model_type: The type of GP model to fit. If not provided, the model will
            be chosen based on the input data only (_get_model); if model_type is
            "saas", the single-task fully-bayesian SAAS model will be used; if
            model_type is "map_saas", the single-task MAP SAAS model will be used.

    Returns:
        A GPyTorchModel (fitted).
    """
    model_type = kwargs.pop("model_type", None)
    if model_type == "saas":
        model = _get_fitted_saas_model(
            train_X=X,
            train_Y=Y,
            train_Yvar=Yvar,
            state_dict=state_dict,
        )
    elif model_type == "map_saas":
        model = get_fitted_map_saas_ensemble(
            train_X=X,
            train_Y=Y,
            train_Yvar=None if Yvar.isnan().any() else Yvar,
        )
    else:
        model = _get_model(
            X=X,
            Y=Y,
            Yvar=Yvar,
            task_feature=task_feature,
            use_input_warping=use_input_warping,
            prior=deepcopy(prior),
            **kwargs,
        )
        if use_loocv_pseudo_likelihood:
            mll_cls = LeaveOneOutPseudoLikelihood
        else:
            mll_cls = ExactMarginalLogLikelihood

        if state_dict is None:
            mll = mll_cls(model.likelihood, model)
            fit_gpytorch_model(mll)
    return model


def _get_fitted_saas_model(
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor,
    state_dict: Optional[Dict[str, Tensor]] = None,
) -> SaasFullyBayesianSingleTaskGP:
    r"""fit a single task SAAS GP for high-dim cases"""
    model = SaasFullyBayesianSingleTaskGP(
        train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar
    )
    if state_dict is None:
        fit_fully_bayesian_model_nuts(
            model,
            warmup_steps=WARMUP_STEPS,
            num_samples=NUM_SAMPLES,
            thinning=THINNING,
            disable_progbar=True,
        )
    return model


def predict_from_mixed_model(model: ModelList, X: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Predicts outcomes given an input tensor and a modellist that is mixed with
    fully-bayesian inference and MAP inference.

    Args:
        model: A batched botorch Model where the batch dimension corresponds
            to sampled hyperparameters.
        X: A `n x d` tensor of input parameters.

    Returns:
        Tensor: The predicted posterior mean as an `n x o`-dim tensor.
        Tensor: The predicted posterior covariance as a `n x o x o`-dim tensor.
    """
    with torch.no_grad():
        posterior = model.posterior(X)
        assert isinstance(
            posterior, PosteriorList
        ), "ModelPredictor is used for ModelList that generates PosteriorList"
        if any(isinstance(p, FullyBayesianPosterior) for p in posterior.posteriors):
            mean = posterior.mean.cpu().detach()
            variance = posterior.variance.cpu().detach().clamp_min(0)
            num_mcmc_samples = mean.shape[MCMC_DIM]
            t1 = variance.sum(dim=MCMC_DIM) / num_mcmc_samples
            t2 = mean.pow(2).sum(dim=MCMC_DIM) / num_mcmc_samples
            t3 = -(mean.sum(dim=MCMC_DIM) / num_mcmc_samples).pow(2)
            variance = t1 + t2 + t3
            mean = mean.mean(dim=MCMC_DIM)
        else:
            mean = posterior.mean.cpu().detach()
            variance = posterior.variance.cpu().detach().clamp_min(0)
        cov = torch.diag_embed(variance)
    return mean, cov


class TargetAwareModel(BotorchModel):
    def __init__(
        self,
        target_metric_config: Dict[str, List[str]],
        model_constructor: TModelConstructor = get_and_fit_online_offline_model,
        model_predictor: TModelPredictor = predict_from_mixed_model,  # pyre-ignore
        acqf_constructor: TAcqfConstructor = get_NEI,
        acqf_optimizer: TOptimizer = scipy_optimizer,  # pyre-ignore
        best_point_recommender: TBestPointRecommender = recommend_best_observed_point,
        refit_on_cv: bool = False,
        refit_on_update: bool = True,
        warm_start_refitting: bool = True,
        prior: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a BoTorch model using target aware GP models.

        Args:
            target_metric_config: A Dict[str, List] that keys are the target metric
                names and values are lists of auxiliary metric names, e.g.
                {"target_metric": ["aux_m1", "aux_m2"]}.
            model_constructor: A callable that instantiates and fits a model on data,
                with signature as described below.
            model_predictor: A callable that predicts using the fitted model, with
                signature as described below.
            acqf_constructor: A callable that creates an acquisition function from a
                fitted model, with signature as described below.
            acqf_optimizer: A callable that optimizes the acquisition function, with
                signature as described below.
            best_point_recommender: A callable that recommends the best point, with
                signature as described below.
            refit_on_cv: If True, refit the model for each fold when performing
                cross-validation.
            refit_on_update: If True, refit the model after updating the training
                data using the `update` method.
            warm_start_refitting: If True, start model refitting from previous
                model parameters in order to speed up the fitting process.
        """
        kwargs["target_metric_config"] = target_metric_config
        BotorchModel.__init__(
            self,
            model_constructor=model_constructor,
            model_predictor=model_predictor,
            acqf_constructor=acqf_constructor,
            acqf_optimizer=acqf_optimizer,
            best_point_recommender=best_point_recommender,
            refit_on_cv=refit_on_cv,
            refit_on_update=refit_on_update,
            warm_start_refitting=warm_start_refitting,
            prior=prior,
            **kwargs,
        )


class TargetAwareMOOModel(MultiObjectiveBotorchModel):
    def __init__(
        self,
        target_metric_config: Dict[str, List[str]],
        model_constructor: TModelConstructor = get_and_fit_online_offline_model,
        model_predictor: TModelPredictor = predict_from_mixed_model,  # pyre-ignore
        acqf_constructor: TAcqfConstructor = get_NEHVI,  # pyre-ignore
        acqf_optimizer: TOptimizer = scipy_optimizer,  # pyre-ignore
        best_point_recommender: TBestPointRecommender = recommend_best_observed_point,
        refit_on_cv: bool = False,
        refit_on_update: bool = True,
        warm_start_refitting: bool = True,
        prior: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a MultiObjective BoTorch model using target aware GP models.

        Args:
            target_metric_config: A Dict[str, List] that keys are the target metric
                names and values are lists of auxiliary metric names, e.g.
                {"target_metric": ["aux_m1", "aux_m2"]}.
            model_constructor: A callable that instantiates and fits a model on data,
                with signature as described below.
            model_predictor: A callable that predicts using the fitted model, with
                signature as described below.
            acqf_constructor: A callable that creates an acquisition function from a
                fitted model, with signature as described below.
            acqf_optimizer: A callable that optimizes the acquisition function, with
                signature as described below.
            best_point_recommender: A callable that recommends the best point, with
                signature as described below.
            refit_on_cv: If True, refit the model for each fold when performing
                cross-validation.
            refit_on_update: If True, refit the model after updating the training
                data using the `update` method.
            warm_start_refitting: If True, start model refitting from previous
                model parameters in order to speed up the fitting process.
        """
        kwargs["target_metric_config"] = target_metric_config
        MultiObjectiveBotorchModel.__init__(
            self,
            model_constructor=model_constructor,
            model_predictor=model_predictor,
            acqf_constructor=acqf_constructor,
            acqf_optimizer=acqf_optimizer,
            best_point_recommender=best_point_recommender,
            refit_on_cv=refit_on_cv,
            refit_on_update=refit_on_update,
            warm_start_refitting=warm_start_refitting,
            prior=prior,
            **kwargs,
        )
