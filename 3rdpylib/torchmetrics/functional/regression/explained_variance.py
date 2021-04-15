# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Sequence, Tuple, Union

import torch
from torch import Tensor

from torchmetrics.utilities.checks import _check_same_shape


def _explained_variance_update(preds: Tensor, target: Tensor) -> Tuple[int, Tensor, Tensor, Tensor, Tensor]:
    _check_same_shape(preds, target)

    n_obs = preds.size(0)
    sum_error = torch.sum(target - preds, dim=0)
    diff = target - preds
    sum_squared_error = torch.sum(diff * diff, dim=0)

    sum_target = torch.sum(target, dim=0)
    sum_squared_target = torch.sum(target * target, dim=0)

    return n_obs, sum_error, sum_squared_error, sum_target, sum_squared_target


def _explained_variance_compute(
    n_obs: Tensor,
    sum_error: Tensor,
    sum_squared_error: Tensor,
    sum_target: Tensor,
    sum_squared_target: Tensor,
    multioutput: str = "uniform_average",
) -> Union[Tensor, Sequence[Tensor]]:
    diff_avg = sum_error / n_obs
    numerator = sum_squared_error / n_obs - (diff_avg * diff_avg)

    target_avg = sum_target / n_obs
    denominator = sum_squared_target / n_obs - (target_avg * target_avg)

    # Take care of division by zero
    nonzero_numerator = numerator != 0
    nonzero_denominator = denominator != 0
    valid_score = nonzero_numerator & nonzero_denominator
    output_scores = torch.ones_like(diff_avg)
    output_scores[valid_score] = 1.0 - (numerator[valid_score] / denominator[valid_score])
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0

    # Decide what to do in multioutput case
    # Todo: allow user to pass in tensor with weights
    if multioutput == "raw_values":
        return output_scores
    if multioutput == "uniform_average":
        return torch.mean(output_scores)
    if multioutput == "variance_weighted":
        denom_sum = torch.sum(denominator)
        return torch.sum(denominator / denom_sum * output_scores)


def explained_variance(
    preds: Tensor,
    target: Tensor,
    multioutput: str = "uniform_average",
) -> Union[Tensor, Sequence[Tensor]]:
    """
    Computes explained variance.

    Args:
        preds: estimated labels
        target: ground truth labels
        multioutput: Defines aggregation in the case of multiple output scores. Can be one
            of the following strings (default is `'uniform_average'`.):

            * `'raw_values'` returns full set of scores
            * `'uniform_average'` scores are uniformly averaged
            * `'variance_weighted'` scores are weighted by their individual variances

    Example:
        >>> from torchmetrics.functional import explained_variance
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> explained_variance(preds, target)
        tensor(0.9572)

        >>> target = torch.tensor([[0.5, 1], [-1, 1], [7, -6]])
        >>> preds = torch.tensor([[0, 2], [-1, 2], [8, -5]])
        >>> explained_variance(preds, target, multioutput='raw_values')
        tensor([0.9677, 1.0000])
    """
    n_obs, sum_error, sum_squared_error, sum_target, sum_squared_target = _explained_variance_update(preds, target)
    return _explained_variance_compute(
        n_obs,
        sum_error,
        sum_squared_error,
        sum_target,
        sum_squared_target,
        multioutput,
    )
