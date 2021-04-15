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
from torchmetrics.classification.accuracy import Accuracy  # noqa: F401
from torchmetrics.classification.auc import AUC  # noqa: F401
from torchmetrics.classification.auroc import AUROC  # noqa: F401
from torchmetrics.classification.average_precision import AveragePrecision  # noqa: F401
from torchmetrics.classification.binned_precision_recall import BinnedAveragePrecision  # noqa: F401
from torchmetrics.classification.binned_precision_recall import BinnedPrecisionRecallCurve  # noqa: F401
from torchmetrics.classification.binned_precision_recall import BinnedRecallAtFixedPrecision  # noqa: F401
from torchmetrics.classification.cohen_kappa import CohenKappa  # noqa: F401
from torchmetrics.classification.confusion_matrix import ConfusionMatrix  # noqa: F401
from torchmetrics.classification.f_beta import F1, FBeta  # noqa: F401
from torchmetrics.classification.hamming_distance import HammingDistance  # noqa: F401
from torchmetrics.classification.hinge import Hinge  # noqa: F401
from torchmetrics.classification.iou import IoU  # noqa: F401
from torchmetrics.classification.matthews_corrcoef import MatthewsCorrcoef  # noqa: F401
from torchmetrics.classification.precision_recall import Precision, Recall  # noqa: F401
from torchmetrics.classification.precision_recall_curve import PrecisionRecallCurve  # noqa: F401
from torchmetrics.classification.roc import ROC  # noqa: F401
from torchmetrics.classification.stat_scores import StatScores  # noqa: F401
