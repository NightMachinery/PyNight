##
#: variance: ddof: "Delta Degrees of Freedom": the divisor used in the calculation is  ``N - ddof``, where ``N`` represents the number of elements. By  default `ddof` is zero.
##
import numpy as np
import torch
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def compute_aopc_lodds(
    *,
    refs,
    orig_probs,
    predicted_probs,
    kfold=10,
    stratified_p=False,
):
    """
    Compute AOPC and LOdds using k-fold cross-validation.

    Parameters:
    refs (list or array): Reference labels for stratification
    orig_probs (torch.Tensor): Original probabilities (batch_size,)
    predicted_probs (torch.Tensor): Predicted probabilities (batch_size,)
    kfold (int): Number of folds for cross-validation (default: 10)
    stratified_p (bool): Whether to use stratified k-fold (default: False)

    Returns:
    dict: Dictionary containing means and variances for AOPC and LOdds
    """
    # Ensure inputs are torch tensors
    orig_probs = torch.as_tensor(orig_probs)
    predicted_probs = torch.as_tensor(predicted_probs)

    # Compute AOPC and LOdds
    aopc = orig_probs - predicted_probs
    lodds = torch.log(orig_probs) - torch.log(predicted_probs)

    # Initialize KFold or StratifiedKFold
    if stratified_p:
        kf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=kfold, shuffle=True, random_state=42)

    # Lists to store fold results
    aopc_means, lodds_means = [], []

    # Perform k-fold cross-validation
    for train_index, test_index in kf.split(aopc, refs):
        aopc_fold = aopc[test_index]
        lodds_fold = lodds[test_index]

        aopc_means.append(torch.mean(aopc_fold).item())
        lodds_means.append(torch.mean(lodds_fold).item())

    # Compute overall statistics
    aopc_var_of_sample_mean = np.var(aopc_means, ddof=0)
    lodds_var_of_sample_mean = np.var(lodds_means, ddof=0)

    # Compute sample statistics
    aopc_sample_mean = torch.mean(aopc).item()
    aopc_sample_var = torch.var(aopc, unbiased=False).item()
    lodds_sample_mean = torch.mean(lodds).item()
    lodds_sample_var = torch.var(lodds, unbiased=False).item()

    return {
        "aopc_mean": aopc_sample_mean,
        "aopc_var": aopc_sample_var,
        "aopc_var_of_sample_mean": aopc_var_of_sample_mean,
        "lodds_mean": lodds_sample_mean,
        "lodds_var": lodds_sample_var,
        "lodds_var_of_sample_mean": lodds_var_of_sample_mean,
    }


###
def cls_metrics_get(
    *,
    refs,
    preds,
    metrics=None,
    kfold=10,
    stratified_p=False,
):
    """
    Compute multiple metrics using k-fold cross-validation.
    Works with both numpy arrays and PyTorch tensors.

    Parameters:
    refs (list, array, or tensor): Reference labels for stratification
    preds (list, array, or tensor): Predicted values
    metrics (dict): Dictionary of metric names and their corresponding functions
    kfold (int): Number of folds for cross-validation (default: 10)
    stratified_p (bool): Whether to use stratified k-fold (default: False)

    Returns:
    dict: Dictionary containing means, variances, and variances of sample means for each metric
    """
    if metrics is None:
        metrics = {
            "accuracy": accuracy_score,
        }

    # Convert inputs to numpy arrays if they're tensors
    if isinstance(refs, torch.Tensor):
        refs = refs.cpu().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()

    # Ensure inputs are numpy arrays
    refs = np.asarray(refs)
    preds = np.asarray(preds)

    # Initialize KFold or StratifiedKFold
    if stratified_p:
        kf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=kfold, shuffle=True, random_state=42)

    # Dictionary to store results for each metric
    results = {metric: [] for metric in metrics}

    # Perform k-fold cross-validation
    for _, test_index in kf.split(preds, refs):
        refs_fold = refs[test_index]
        preds_fold = preds[test_index]

        # Compute each metric for this fold
        for metric_name, metric_func in metrics.items():
            metric_value = metric_func(refs_fold, preds_fold)
            results[metric_name].append(metric_value)

    # Compute overall statistics
    final_results = {}
    for metric_name, metric_values in results.items():
        metric_values = np.array(metric_values)
        final_results[f"{metric_name}_mean"] = np.mean(metric_values)
        final_results[f"{metric_name}_var"] = np.var(metric_values, ddof=0)
        final_results[f"{metric_name}_var_of_sample_mean"] = np.var(
            metric_values, ddof=0
        )

    return final_results


###
