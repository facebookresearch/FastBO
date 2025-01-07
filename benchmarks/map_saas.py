#!/usr/bin/env python3

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from botorch.exceptions import UnsupportedError
from botorch.fit import fit_gpytorch_mll
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch import settings
from gpytorch.constraints import Interval
from gpytorch.kernels import AdditiveKernel, Kernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior, HalfCauchyPrior, NormalPrior
from torch import Tensor
from torch.distributions.half_cauchy import HalfCauchy
from torch.nn import Parameter


EPS = 1e-8


class LogTransformedInterval(Interval):
    """Modification of the GPyTorch interval class.

    The Interval class in GPyTorch will map the parameter to the range [0, 1] before
    applying the inverse transform. We don't want to do this when using log as an
    inverse transform. This class will skip this step and apply the log transform
    directly to the parameter values so we can optimize log(parameter) under the bound
    constraints log(lower) <= log(parameter) <= log(upper).
    """

    def __init__(self, lower_bound, upper_bound, initial_value=None):
        super().__init__(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            transform=torch.exp,
            inv_transform=torch.log,
            initial_value=initial_value,
        )

        # Save the untransformed initial value
        self.register_buffer(
            "initial_value_untransformed",
            torch.tensor(initial_value).to(self.lower_bound)
            if initial_value is not None
            else None,
        )

        if settings.debug.on():
            max_bound = torch.max(self.upper_bound)
            min_bound = torch.min(self.lower_bound)
            if max_bound == math.inf or min_bound == -math.inf:
                raise RuntimeError(
                    "Cannot make an Interval directly with non-finite bounds. Use a "
                    "derived class like GreaterThan or LessThan instead."
                )

    def transform(self, tensor):
        if not self.enforced:
            return tensor

        transformed_tensor = self._transform(tensor)
        return transformed_tensor

    def inverse_transform(self, transformed_tensor):
        if not self.enforced:
            return transformed_tensor

        tensor = self._inv_transform(transformed_tensor)
        return tensor


class SaasPriorHelper:
    """Helper class for specifying parameter and setting closures."""

    def __init__(self, tau: Optional[float] = None):
        self._tau = torch.as_tensor(tau) if tau is not None else None

    def tau(self, m):
        return (
            self._tau.to(m.lengthscale)
            if self._tau is not None
            else m.raw_tau_constraint.transform(m.raw_tau)
        )

    def inv_lengthscale_prior_param_or_closure(self, m):
        tau = self.tau(m)
        return tau.view(*tau.shape, 1, 1) / (m.lengthscale**2)

    def inv_lengthscale_prior_setting_closure(self, m, value):
        lb = m.raw_lengthscale_constraint.lower_bound
        ub = m.raw_lengthscale_constraint.upper_bound
        tau = self.tau(m).unsqueeze(-1)
        m._set_lengthscale((tau / value.to(tau)).sqrt().clamp(lb + EPS, ub - EPS))

    def tau_prior_param_or_closure(self, m):
        return m.raw_tau_constraint.transform(m.raw_tau)

    def tau_prior_setting_closure(self, m, value):
        lb = m.raw_tau_constraint.lower_bound
        ub = m.raw_tau_constraint.upper_bound
        m.raw_tau.data.fill_(
            m.raw_tau_constraint.inverse_transform(
                value.clamp(lb + EPS, ub - EPS)
            ).item()
        )


def add_saas_prior(
    base_kernel: Kernel, tau: Optional[float] = None, log_scale: bool = True, **tkwargs
) -> Kernel:
    """Add a SAAS prior to a given base_kernel.

    The SAAS prior is given by tau / lengthscale^2 ~ HC(1.0). If tau is None,
    we place an additional HC(0.1) prior on tau similar to the original SAAS prior
    that relies on inference with NUTS.

    Args:
        base_kernel: Base kernel that has a lengthscale and uses ARD.
            Note that this function modifies the kernel object in place.
        tau: Value of the global shrinkage. If `None`, infer the global
            shrinkage parameter.
        log_scale: Set to `True` if the lengthscale and tau should be optimized on
            a log-scale without any domain rescaling. That is, we will learn
            `raw_lengthscale := log(lengthscale)` and this hyperparameter needs to
            satisfy the corresponding bound constraints. Setting this to `True` will
            generally improve the numerical stability, but requires an optimizer that
            can handle bound constraints, e.g., L-BFGS-B.

    Returns:
        Base kernel with SAAS priors added.

    Example:
        >>> matern_kernel = MaternKernel(...)
        >>> add_saas_prior(matern_kernel, tau=None)  # Add a SAAS prior
    """
    if not base_kernel.has_lengthscale:
        raise UnsupportedError("base_kernel must have lengthscale(s)")
    if hasattr(base_kernel, "lengthscale_prior"):
        raise UnsupportedError("base_kernel must not specify a lengthscale prior")
    tkwargs = tkwargs or {"device": base_kernel.device, "dtype": base_kernel.dtype}

    batch_shape = base_kernel.raw_lengthscale.shape[:-2]
    IntervalClass = LogTransformedInterval if log_scale else Interval
    base_kernel.register_constraint(
        param_name="raw_lengthscale",
        constraint=IntervalClass(0.01, 1e4, initial_value=1),
        replace=True,
    )
    prior_helper = SaasPriorHelper(tau=tau)
    if tau is None:  # Place a HC(0.1) prior on tau
        base_kernel.register_parameter(
            name="raw_tau",
            parameter=Parameter(torch.full(batch_shape, 0.1, **tkwargs)),
        )
        base_kernel.register_constraint(
            param_name="raw_tau",
            constraint=IntervalClass(1e-3, 10, initial_value=0.1),
            replace=True,
        )
        base_kernel.register_prior(
            name="tau_prior",
            prior=HalfCauchyPrior(torch.tensor(0.1, **tkwargs)),
            param_or_closure=prior_helper.tau_prior_param_or_closure,
            setting_closure=prior_helper.tau_prior_setting_closure,
        )
    # Place a HC(1) prior on tau / lengthscale^2
    base_kernel.register_prior(
        name="inv_lengthscale_prior",
        prior=HalfCauchyPrior(torch.tensor(1.0, **tkwargs)),
        param_or_closure=prior_helper.inv_lengthscale_prior_param_or_closure,
        setting_closure=prior_helper.inv_lengthscale_prior_setting_closure,
    )
    return base_kernel


def _get_map_saas_model(
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Optional[Tensor] = None,
    input_transform: Optional[InputTransform] = None,
    outcome_transform: Optional[OutcomeTransform] = None,
    tau: Optional[float] = None,
) -> Union[FixedNoiseGP, SingleTaskGP]:
    """Helper method for creating an unfitted MAP SAAS model."""
    # TODO: Shape checks
    tkwargs = {"device": train_X.device, "dtype": train_X.dtype}
    _, aug_batch_shape = SingleTaskGP.get_batch_dimensions(
        train_X=train_X, train_Y=train_Y
    )
    mean_module = get_mean_module_with_normal_prior(batch_shape=aug_batch_shape)
    if input_transform is not None:
        with torch.no_grad():
            transformed_X = input_transform(train_X)
        ard_num_dims = transformed_X.shape[-1]
    else:
        ard_num_dims = train_X.shape[-1]
    base_kernel = MaternKernel(
        nu=2.5, ard_num_dims=ard_num_dims, batch_shape=aug_batch_shape
    )
    add_saas_prior(base_kernel=base_kernel, tau=tau, **tkwargs)
    covar_module = ScaleKernel(
        base_kernel=base_kernel,
        outputscale_constraint=LogTransformedInterval(1e-2, 1e4, initial_value=10),
        batch_shape=aug_batch_shape,
    )
    if train_Yvar is not None:
        return FixedNoiseGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            mean_module=mean_module,
            covar_module=covar_module,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
    likelihood = get_gaussian_likelihood_with_gamma_prior(batch_shape=aug_batch_shape)
    return SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        mean_module=mean_module,
        covar_module=covar_module,
        likelihood=likelihood,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
    )


def get_fitted_map_saas_model(
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Optional[Tensor] = None,
    input_transform: Optional[InputTransform] = None,
    outcome_transform: Optional[OutcomeTransform] = None,
    tau: Optional[float] = None,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[FixedNoiseGP, SingleTaskGP]:
    """Get a fitted MAP SAAS model with a Matern kernel.

    Args:
        train_X: Tensor of shape `n x d` with training inputs.
        train_Y: Tensor of shape `n x 1` with training targets.
        train_Yvar: Optional tensor of shape `n x 1` with observed noise,
            inferred if None.
        input_transform: An optional input transform.
        outcome_transform: An optional outcome transforms.
        tau: Fixed value of the global shrinkage tau. If None, the model
            places a HC(0.1) prior on tau.
        optimizer_kwargs: A dict of options for the optimizer passed
            to fit_gpytorch_mll.

    Returns:
        A fitted SingleTaskGP with a Matern kernel.
    """

    # make sure optimizer_kwargs is a Dict
    optimizer_kwargs = optimizer_kwargs or {}

    model = _get_map_saas_model(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        input_transform=input_transform.train()
        if input_transform is not None
        else None,
        outcome_transform=outcome_transform,
        tau=tau,
    )
    mll = ExactMarginalLogLikelihood(model=model, likelihood=model.likelihood)
    fit_gpytorch_mll(mll, optimizer_kwargs=optimizer_kwargs)
    return model


def get_fitted_map_saas_ensemble(
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Optional[Tensor] = None,
    input_transform: Optional[InputTransform] = None,
    outcome_transform: Optional[OutcomeTransform] = None,
    taus: Optional[Union[Tensor, List[float]]] = None,
    num_taus: int = 4,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
) -> SaasFullyBayesianSingleTaskGP:
    """Get a fitted SAAS ensemble using several different tau values.

    Args:
        train_X: Tensor of shape `n x d` with training inputs.
        train_Y: Tensor of shape `n x 1` with training targets.
        train_Yvar: Optional tensor of shape `n x 1` with observed noise,
            inferred if None.
        input_transform: An optional input transform.
        outcome_transform: An optional outcome transforms.
        taus: Global shrinkage values to use. If None, we sample `num_taus` values
            from an HC(0.1) distrbution.
        num_taus: Optional argument for how many taus to sample.
        optimizer_kwargs: A dict of options for the optimizer passed
            to fit_gpytorch_mll.

    Returns:
        A fitted SaasFullyBayesianSingleTaskGP with a Matern kernel.
    """
    tkwargs = {"device": train_X.device, "dtype": train_X.dtype}
    if taus is None:
        taus = HalfCauchy(0.1).sample([num_taus]).to(**tkwargs)
    num_samples = len(taus)
    if num_samples == 1:
        raise ValueError(
            "Use `get_fitted_map_saas_model` if you only specify one value of tau"
        )

    mean = torch.zeros(num_samples, **tkwargs)
    outputscale = torch.zeros(num_samples, **tkwargs)
    lengthscale = torch.zeros(num_samples, train_X.shape[-1], **tkwargs)
    noise = torch.zeros(num_samples, **tkwargs)

    # Fit a model for each tau and save the hyperparameters
    for i, tau in enumerate(taus):
        model = get_fitted_map_saas_model(
            train_X,
            train_Y,
            train_Yvar=train_Yvar,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            tau=tau,
            optimizer_kwargs=optimizer_kwargs,
        )
        mean[i] = model.mean_module.constant.detach().clone()
        outputscale[i] = model.covar_module.outputscale.detach().clone()
        lengthscale[i, :] = model.covar_module.base_kernel.lengthscale.detach().clone()
        if train_Yvar is None:
            noise[i] = model.likelihood.noise.detach().clone()

    # Load the samples into a fully Bayesian SAAS model
    ensemble_model = SaasFullyBayesianSingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        input_transform=input_transform.train()
        if input_transform is not None
        else None,
        outcome_transform=outcome_transform,
    )
    mcmc_samples = {
        "mean": mean,
        "outputscale": outputscale,
        "lengthscale": lengthscale,
    }
    if train_Yvar is None:
        mcmc_samples["noise"] = noise
    ensemble_model.train()
    ensemble_model.load_mcmc_samples(mcmc_samples=mcmc_samples)
    ensemble_model.eval()
    return ensemble_model


def get_mean_module_with_normal_prior(
    batch_shape: Optional[torch.Size] = None,
) -> ConstantMean:
    """Return constant mean with a N(0, 1) prior constrained to [-10, 10].

    This prior assumes the outputs (targets) have been standardized to have zero mean
    and unit variance.

    Args:
        batch_shape: Optional batch shape for the constant-mean module.

    Returns:
        ConstantMean module.
    """
    return ConstantMean(
        constant_prior=NormalPrior(loc=0.0, scale=1.0),
        constant_constraint=Interval(
            -10,
            10,
            initial_value=0,
            transform=None,
        ),
        batch_shape=batch_shape or torch.Size(),
    )


def get_gaussian_likelihood_with_gamma_prior(batch_shape: Optional[torch.Size] = None):
    """Return Gaussian likelihood with a Gamma(0.9, 10) prior.

    This prior prefers small noise, but also has heavy tails.

    Args:
        batch_shape: Batch shape for the likelihood.

    Returns:
        GaussianLikelihood with Gamma(0.9, 10) prior constrained to [1e-4, 0.1].
    """
    return GaussianLikelihood(
        noise_prior=GammaPrior(0.9, 10.0),
        noise_constraint=LogTransformedInterval(1e-4, 1, initial_value=1e-2),
        batch_shape=batch_shape or torch.Size(),
    )


def get_additive_map_saas_covar_module(
    ard_num_dims: int,
    num_taus: int = 4,
    active_dims: Optional[Tuple[int, ...]] = None,
    batch_shape: Optional[torch.Size] = None,
):
    """Return an additive map SAAS covar module.

    The constructed kernel is an additive kernel with `num_taus` terms. Each term is a
    scaled Matern kernel with a SAAS prior and a tau sampled from a HalfCauchy(0, 1)
    distrbution.

    Args:
        ard_num_dims: The number of inputs dimensions.
        num_taus: The number of taus to use (4 if omitted).
        active_dims: Active dims for the covar module. The kernel will be evaluated
            only using these columns of the input tensor.
        batch_shape: Batch shape for the covar module.

    Returns:
        An additive MAP SAAS covar module.
    """
    batch_shape = batch_shape or torch.Size()
    kernels = []
    for _ in range(num_taus):
        base_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            active_dims=active_dims,
        )
        add_saas_prior(base_kernel=base_kernel, tau=HalfCauchy(0.1).sample(batch_shape))
        scaled_kernel = ScaleKernel(
            base_kernel=base_kernel,
            outputscale_constraint=LogTransformedInterval(1e-2, 1e4, initial_value=10),
            batch_shape=batch_shape,
        )
        kernels.append(scaled_kernel)
    return AdditiveKernel(*kernels)


class AdditiveMapSaasSingleTaskGP(SingleTaskGP):
    """An additive MAP SAAS single-task GP.

    This is a maximum-a-posteriori (MAP) version of sparse axis-aligned subspace BO
    (SAASBO), see `SaasFullyBayesianSingleTaskGP` for more details. SAASBO is a
    high-dimensional Bayesian optimization approach that uses approximate fully
    Bayesian inference via NUTS to learn the model hyperparameters. This works very
    well, but is very computationally expensive which limits the use of SAASBO to a
    small (~100) number of trials. Two of the main benefits with SAASBO are:
        1. A sparse prior on the inverse lengthscales that avoid overfitting
        2. The ability to sample several (~16) sets of hyperparameters from the
        posterior that we can average over when computing the acquisition
        function (ensembling).

    The goal of this Additive MAP SAAS model is to retain the main benefits of the SAAS
    model while significantly speeding up the time to fit the model. We achieve this by
    creating an additive kernel where each kernel in the sum is a Matern-5/2 kernel
    with a SAAS prior and a separate outputscale. The sparsity level for each kernel
    is sampled from an HC(0.1) distribution leading to a mix of sparsity levels (as is
    often the case for the fully Bayesian SAAS model). We learn all the hyperparameters
    using MAP inference which is significantly faster than using NUTS.

    While we often find that the original SAAS model with NUTS performs better, the
    additive MAP SAAS model can be several orders of magnitude faster to fit, which
    makes it applicable to problems with potentially thousands of trials.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
        num_taus: int = 4,
    ) -> None:
        """
        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            train_Yvar: A `batch_shape x n x m` tensor of observed noise.
            outcome_transform: An optional outcome transform.
            input_transform: An optional input transform.
            num_taus: The number of taus to use (4 if omitted).
        """
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        mean_module = get_mean_module_with_normal_prior(
            batch_shape=self._aug_batch_shape
        )
        if train_Yvar is not None:
            _, _, train_Yvar = self._transform_tensor_args(
                X=train_X, Y=train_Y, Yvar=train_Yvar
            )
        likelihood = (
            FixedNoiseGaussianLikelihood(
                noise=train_Yvar, batch_shape=self._aug_batch_shape
            )
            if train_Yvar is not None
            else get_gaussian_likelihood_with_gamma_prior(
                batch_shape=self._aug_batch_shape
            )
        )
        covar_module = get_additive_map_saas_covar_module(
            ard_num_dims=train_X.shape[-1],
            num_taus=num_taus,
            batch_shape=self._aug_batch_shape,
        )

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
