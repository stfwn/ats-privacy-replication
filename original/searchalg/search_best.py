import argparse
import json
from pathlib import Path
import numpy as np


def find_best(dataset_name: str, model_name: str, thresh_acc: int = -85, n: int = 10):
    """
    Finds list of best augmentation sets. It's paper section 4.4. Also saves that list
    :params data_name: name of the dataset. e.g. cifar100
    :params arch_name: name of model's architecture, e.g. resnet20
    :param thresh_acc: Accuracy Score Threshold
    :params n: Maximum number of policies
    :returns: List of best policies sets, sorted by S_pri.
    """

    log_dir = Path('logs/{}-{}/augmentations'.format(dataset_name, model_name))

    results = []
    for aug_results_path in log_dir.iterdir():
        if aug_results_path.is_file():
            with open(str(aug_results_path), 'r') as f:
                loaded_results = json.load(f)
            search_mean = np.mean(loaded_results['S_pri'])
            accuracy_score_mean = np.mean(loaded_results['accuracy'])
            results.append((aug_results_path, search_mean, accuracy_score_mean))
    num_all = len(results)
    print(f"Starting sorting the result by search mean...")
    # sort the results by the search mean
    results.sort(key=lambda x: x[1])
    # sort by accuracy score mean
    print(f"Starting the accuracy score filtering, threshold is: {thresh_acc}...")
    for idx, result in enumerate(results):
        if result[2] < thresh_acc:
            results.pop(idx)
    # this might give info if this is even efective
    print(f"{num_all-len(results)} policy sets were below threshold")
    results = results[:n]
    best_path = log_dir.parent / "best_results_search.json"
    best_results = {}
    for idx, result in enumerate(results):
        best_results[idx] = {"auglist": result[0].name[:-5], "search_mean": result[1],
                             "accuracy_score_mean": result[2]}
    print(f"Best results: {best_results}")
    with open(best_path, 'wa') as f:
        json.dump(best_results, f)
    return best_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')
    parser.add_argument('--arch', default=None, required=True, type=str, help='Vision model.')
    parser.add_argument('--data', default=None, required=True, type=str, help='Vision dataset.')
    parser.add_argument('--thresh-acc', default=-85, required=False, type=int, help='Accuracy Score Threshold')
    parser.add_argument('--n', default=10, required=False, type=int, help='Maximum number of policies')
    opt = parser.parse_args()
    best_results = find_best(opt.data, opt.arch)
