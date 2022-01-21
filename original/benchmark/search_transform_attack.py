import json
import os
import torch
import numpy as np
import torch.nn.functional as F
import copy
import time
from math import inf

from original.benchmark.comm import create_model, preprocess
from original import inversefed


def get_batch_jacobian(model, x, target):
    """
    Returns gradient of net prediction on x and detached target
    :param model: model that is being evaluated
    :param x: input image for evaluation
    :param target: target label
    """
    model.eval()
    model.zero_grad()
    x.requires_grad_(True)
    # this is prediction
    y = model(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach()


def calculate_gradient(model, inputs, labels, loss_fn):
    """
    Calculates gradient of model prediction on input.
    :param model: model that predicts
    :param inputs: inputs to be classified
    :param labels: labels of inputs
    :param loss_fn: loss function
    :returns: gradient
    """
    model.zero_grad()
    target_loss, _, _ = loss_fn(model(inputs), labels)
    dw = torch.autograd.grad(target_loss, model.parameters())
    return dw


def cal_dis(a, b, metric='L2'):
    """
    This function calls the metric with flattened a and b.
    Metric should be given as a  string:
    'L2' - L2 norm
    'L1' - L1 norm
    'cos' - cosine similarity
    """
    a, b = a.flatten(), b.flatten()
    if metric == 'L2':
        return torch.mean((a - b) * (a - b)).item()
    elif metric == 'L1':
        return torch.mean(torch.abs(a - b)).item()
    elif metric == 'cos':
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    else:
        raise NotImplementedError


def prepare_data(idx_list, loader, setup, max_num=inf):
    """
    Function used for getting images and labels from dataset
    :param idx_list: list of data indices that should be prepared
    :param loader: data loader
    :param setup: setup object created by inversefed.utils.system_startup function
    :param max_num: max number of items in the prepared data
    :returns: prepared images, prepared labels
    """
    images, labels = [], []
    for idx in idx_list:
        img, label = loader.dataset[idx]
        if label not in labels:
            labels.append(torch.as_tensor((label,), device=setup['device']))
            images.append(img.to(**setup))
        if len(labels) >= max_num:
            break
    images = torch.stack(images)
    labels = torch.cat(labels)
    return images, labels


def accuracy_score_metric(idx_list, model, validloader, setup, k=1e-5):
    """
    Implementation of 4.3 part of the paper. Calculated the accuracy score metric.
    :param idx_list: list of data indices that should be prepared
    :param model: model for evaluation
    :param validloader: data loader
    :param setup: setup object created by inversefed.utils.system_startup function
    :param k: parameter epsilon for numerical stability from equation (10)
    :returns: accuracy score of given model and augmentations
    """
    # get the data
    images, labels = prepare_data(idx_list, validloader, setup)
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
    return -np.sum(np.log(eigenvalues + k) + 1. / (eigenvalues + k))


def reconstruct(idx, model, loss_fn, trainloader, validloader, setup, opt):
    """
    Implementation of part 4.2 of the paper
    """
    if opt.data == 'cifar100':
        dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    elif opt.data == 'FashionMinist':
        dm = torch.Tensor([0.1307]).view(1, 1, 1).to(setup['device'])
        ds = torch.Tensor([0.3081]).view(1, 1, 1).to(setup['device'])
    else:
        raise NotImplementedError

    images, labels = prepare_data(list(range(idx, idx + len(validloader))), validloader, setup, opt.num_images)
    model.zero_grad()

    # calcuate original dW (gradient)
    original_loss, _, _ = loss_fn(model(images), labels)
    input_gradient = torch.autograd.grad(original_loss, model.parameters())

    # attack model
    model.eval()
    dw_list = []
    bin_num = 20
    noise_input = (torch.rand((images.shape)).to(setup['device']) - dm) / ds
    for dis_iter in range(bin_num + 1):
        model.zero_grad()
        reconstructed_image = (1.0 / bin_num * dis_iter * images + 1. / bin_num * (
                bin_num - dis_iter) * noise_input).detach()
        reconstructed_image_gradient = calculate_gradient(model, reconstructed_image, labels, loss_fn)
        dw_loss = sum([cal_dis(dw_a, dw_b, metric='cos') for dw_a, dw_b in
                       zip(reconstructed_image_gradient, input_gradient)]) / len(
            input_gradient)
        dw_list.append(dw_loss)

    interval_distance = cal_dis(noise_input, images, metric='L1') / bin_num

    return area_ratio(dw_list, interval_distance, bin_num), dw_list


def area_ratio(y_list, inter, bin_num):
    area = 0
    max_area = inter * bin_num
    for idx in range(1, len(y_list)):
        prev = y_list[idx - 1]
        cur = y_list[idx]
        area += (prev + cur) * inter / 2
    return area / max_area


def main(opt):
    print(f"Testing policy '{opt.aug_list}'...")
    pathname = 'logs/{}-{}/augmentations/{}.json'.format(opt.data, opt.arch, opt.aug_list)
    if os.path.exists(pathname):
        print(f"Results for {opt.aug_list} already exists, skipping")
        return

    # init env
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative')
    defs.epochs = opt.epochs

    loss_fn, trainloader, validloader = preprocess(opt, defs, valid=True)
    model = create_model(opt)
    model.to(**setup)
    old_state_dict = copy.deepcopy(model.state_dict())
    model.load_state_dict(torch.load(opt.model_checkpoint, map_location=setup["device"]))

    model.eval()

    # at this step model should be loaded from checkpoint and set to eval
    metric_list = []
    gradsims = []
    start = time.time()
    # sample indices of the data???
    sample_list = [200 + i * 5 for i in range(100)]

    #  for each sample(?) get metrics from reconstruction
    for attack_id, idx in enumerate(sample_list):
        metric, dw_list = reconstruct(idx, model, loss_fn, trainloader, validloader, setup, opt)
        metric_list.append(metric)
        gradsims.append(dw_list)
        print('attach {}th in {}, metric {}'.format(attack_id, opt.aug_list, metric))

    root_dir = os.path.dirname(pathname)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    results = {}

    results["grad_sim"] = gradsims
    if len(metric_list) > 0:
        print("Mean of metric list:", np.mean(metric_list))
        results["S_pri"] = list(metric_list)

    # maybe need old_state_dict
    model.load_state_dict(old_state_dict)
    score_list = list()
    for run in range(10):
        large_samle_list = [200 + run * 100 + i for i in range(100)]
        score = accuracy_score_metric(large_samle_list, model, validloader, setup)
        score_list.append(score)

    print('time cost ', time.time() - start)

    results["accuracy"] = score_list
    print(score_list)

    with open(pathname, "w") as f:
        json.dump(results, f)
