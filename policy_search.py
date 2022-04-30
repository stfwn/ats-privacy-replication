import json
import random
import numpy as np
import torch.cuda
import models
import data_modules
import torch.nn.functional as F

from tqdm.auto import tqdm
from pathlib import Path
from math import inf
from joblib import Parallel, delayed
from queue import Queue
from argparse import Namespace
from filelock import FileLock

from utils import join_augmentations
from policy import NUM_TRANSFORMATIONS, EMPTY_TRANSFORMATION


def parallel_policy_search(
    num_schemes: int,
    model: str,
    data: str,
    epochs: int,
    model_checkpoint: str,
    num_transform: int = 3,
    num_per_gpu: int = 20,
    num_images: int = 1,
    schemes: list = None,
    data_dir: str = "data/",
):
    gpu_queue = Queue()
    num_gpu = max(torch.cuda.device_count(), 1)
    for gpu_ids in range(num_gpu):
        for _ in range(num_per_gpu):
            gpu_queue.put(gpu_ids)

    if schemes is None:
        schemes = create_schemes(num_schemes, num_transform)
    Parallel(n_jobs=num_gpu * num_per_gpu, require="sharedmem")(
        delayed(search_transform_attack_handler)(
            scheme,
            model,
            data,
            epochs,
            model_checkpoint,
            num_images,
            data_dir,
            gpu_queue,
        )
        for scheme in schemes
    )


def search_transform_attack_handler(
    scheme,
    model,
    data,
    epochs,
    model_checkpoint,
    num_images,
    data_dir,
    gpu_queue,
):
    gpu = gpu_queue.get()
    try:
        args = Namespace(
            **dict(
                aug_list=scheme,
                model=model,
                dataset=data,
                epochs=epochs,
                num_images=num_images,
                model_checkpoint=model_checkpoint,
                data_dir=data_dir,
            )
        )
        if torch.cuda.is_available():
            with torch.cuda.device(gpu):
                search_transform_attack(args)
        else:
            search_transform_attack(args)
    finally:
        gpu_queue.put(gpu)


def search_transform_attack(args):
    print(f"Testing policy '{args.aug_list}'...")
    logfile = (
        Path("logs")
        / f"{args.dataset}-{args.model}"
        / "augmentations"
        / f"{join_augmentations([args.aug_list])}.json"
    )
    if logfile.exists():
        print(f"Results for {args.aug_list} already exists, skipping")
        return
    logfile.parent.mkdir(exist_ok=True, parents=True)

    # Prepare model, data and loss
    with FileLock(".model_load.lock"):  # avoid problems in Pytorch Lightning
        model = (
            models.get_by_name(args.model)
            .load_from_checkpoint(checkpoint_path=args.model_checkpoint)
            .to("cuda" if torch.cuda.is_available() else "cpu")
        )
    model.eval()
    data_module = data_modules.get_by_name(args.dataset)(
        data_dir=args.data_dir,
        policy_list=[args.aug_list],
        apply_policy_to_test=True,
    )
    data_module.prepare_data()
    data_module.setup(stage="test")
    data_mean = data_module.mean.view(data_module.num_channels, 1, 1).to(
        model.device
    )
    data_std = data_module.std.view(data_module.num_channels, 1, 1).to(
        model.device
    )

    spri_img_indices = list(
        range(200, 700, 5)
    )  # indices from original implementation

    # Calculate S_pri
    s_pris = []
    gradsims = []
    print("Calculating S_pri...")
    for img_idx in tqdm(spri_img_indices, unit="img"):
        img, label = prepare_data([img_idx], data_module, model.device)
        s_pri, dw_list = reconstruct(
            model, img, label, data_mean=data_mean, data_std=data_std
        )
        s_pris.append(s_pri)
        gradsims.append(dw_list)
    print(f"Attack '{args.aug_list}' has average S_pri={np.mean(s_pris):.2f}")
    del model

    # Calculate Accuracy
    # initialize model with random weights
    untrained_model = models.get_by_name(args.model)(
        num_channels=data_module.num_channels,
        num_classes=data_module.num_classes,
        epochs=1,  # not used
    )
    untrained_model.to("cuda" if torch.cuda.is_available() else "cpu")
    untrained_model.eval()

    accuracies = []
    for run in range(10):
        large_sample_list = [200 + run * 100 + i for i in range(100)]
        images, labels = prepare_data(
            large_sample_list, data_module, untrained_model.device
        )
        accuracy = accuracy_score_metric(untrained_model, images, labels)
        accuracies.append(accuracy)
    print(
        f"Attack '{args.aug_list}' has average S_acc={np.mean(accuracies):.2f}"
    )

    results = {"gradsim": gradsims, "S_pri": s_pris, "accuracy": accuracies}
    with open(logfile, "w") as f:
        json.dump(results, f)


def reconstruct(
    model, img, label, steps: int = 21, data_mean=0.0, data_std=1.0
):
    input_gradient = calculate_gradient(model, img, label)

    dw_list = []
    noise_input = (
        torch.rand(img.shape).to(model.device) - data_mean
    ) / data_std
    for reconstruct_img in interpolate(noise_input, img, steps):
        reconstructed_gradient = calculate_gradient(
            model, reconstruct_img, label
        )
        dw_loss = np.mean(
            [
                cosine_similarity(dw_rec, dw_inp)
                for dw_rec, dw_inp in zip(
                    reconstructed_gradient, input_gradient
                )
            ]
        )
        dw_list.append(dw_loss)

    auc = np.trapz(dw_list, dx=1 / (steps - 1))
    return auc, dw_list


def cosine_similarity(input1, input2):
    input1, input2 = input1.flatten(), input2.flatten()
    return F.cosine_similarity(input1, input2, dim=0).item()


def interpolate(a, b, steps: int = 20):
    for alpha in np.linspace(1, 0, steps):
        yield a * alpha + b * (1 - alpha)


def prepare_data(idx_list, loader, device=None, max_num=inf):
    """
    Function used for getting images and labels from dataset
    :param idx_list: list of data indices that should be prepared
    :param loader: data loader
    :param device: where to send the output to
    :param max_num: max number of items in the prepared data
    :returns: prepared images, prepared labels
    """
    images, labels = [], []
    for idx in idx_list:
        img, label = loader.test[idx]
        if label not in labels:
            labels.append(torch.as_tensor((label,)))
            images.append(img)
        if len(labels) >= max_num:
            break

    images = torch.stack(images)
    labels = torch.cat(labels)
    if device is not None:
        images = images.to(device)
        labels = labels.to(device)

    return images, labels


def calculate_gradient(model, inputs, labels):
    """
    Calculates gradient of model prediction on input.
    :param model: model that predicts
    :param inputs: inputs to be classified
    :param labels: labels of inputs
    :returns: gradient
    """
    model.zero_grad()
    target_loss = model.loss_function(model(inputs), labels)
    dw = torch.autograd.grad(target_loss, model.parameters())
    return dw


def accuracy_score_metric(model, images, labels, k=1e-5):
    """
    Implementation of 4.3 part of the paper. Calculated the accuracy score metric.
    :param model: model for evaluation
    :param images: images to compute the metric on
    :param labels: corresponding labels to compute the metric on
    :param k: parameter epsilon for numerical stability from equation (10)
    :returns: accuracy score of given model and augmentations
    """
    # get the data
    model.zero_grad()
    # calculate the Gradient Jacobian matrix, equation (8)
    jacobs, labels = get_batch_jacobian(model, images, labels)
    jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
    # compute correlation matrix - equation (9)
    correlation_matrix = np.corrcoef(jacobs)
    # get eigenvalues of sum of jacobians
    eigenvalues, _ = np.linalg.eig(correlation_matrix)
    # calculate accuracy score - equation (10)
    # why here is minus?? how does it influence filtering by threshold
    return -np.sum(np.log(eigenvalues + k) + 1.0 / (eigenvalues + k))


def get_batch_jacobian(model, img, label):
    """
    Returns gradient of net prediction on x and detached target
    :param model: model that is being evaluated
    :param img: input image for evaluation
    :param label: target label
    """
    model.eval()
    model.zero_grad()
    img.requires_grad_(True)
    # this is prediction
    y = model(img)
    y.backward(torch.ones_like(y))
    jacob = img.grad.detach()
    return jacob, label.detach()


def create_schemes(num_schemes: int, num_transform: int = 3):
    schemes = []
    for _ in range(num_schemes):
        scheme = [random_transformation() for _ in range(num_transform)]
        scheme = [t for t in scheme if t != EMPTY_TRANSFORMATION]
        schemes.append(scheme)
    return schemes


def random_transformation():
    return random.randint(EMPTY_TRANSFORMATION, NUM_TRANSFORMATIONS - 1)


def find_best(
    dataset_name: str, model_name: str, thresh_acc: int = -85, n: int = 10
):
    """
    Finds list of best augmentation sets. It's paper section 4.4. Also saves that list
    :param dataset_name: name of the dataset. e.g. cifar100
    :param model_name: name of model's architecture, e.g. resnet20
    :param thresh_acc: Accuracy Score Threshold
    :param n: Maximum number of policies
    :returns: List of best policies sets, sorted by S_pri.
    """

    log_dir = Path("logs/{}-{}/augmentations".format(dataset_name, model_name))

    results = []
    for aug_results_path in log_dir.iterdir():
        if aug_results_path.is_file():
            with open(str(aug_results_path), "r") as f:
                loaded_results = json.load(f)
            search_mean = np.mean(loaded_results["S_pri"])
            accuracy_score_mean = np.mean(loaded_results["accuracy"])
            results.append((aug_results_path, search_mean, accuracy_score_mean))
    num_all = len(results)
    print(f"Starting sorting the result by search mean...")
    # sort the results by the search mean
    results.sort(key=lambda x: x[1])
    # sort by accuracy score mean
    print(
        f"Starting the accuracy score filtering, threshold is: {thresh_acc}..."
    )
    results = [result for result in results if result[2] >= thresh_acc]
    # this might give info if this is even effective
    print(f"{num_all - len(results)} policy sets were below threshold")
    results = results[:n]
    best_path = log_dir.parent / "best_results_search.json"
    best_results = {}
    for idx, result in enumerate(results):
        best_results[idx] = {
            "auglist": result[0].name[:-5],
            "search_mean": result[1],
            "accuracy_score_mean": result[2],
        }
    print(f"Best results: {best_results}")
    with open(best_path, "w") as f:
        json.dump(best_results, f)
    return best_results
