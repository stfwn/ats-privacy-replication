import os
import torch
import numpy as np
import torch.nn.functional as F
import copy
import time

from math import inf
from original.benchmark.comm import create_model, preprocess
from original import inversefed


def eval_score(jacob):
    corrs = np.corrcoef(jacob)
    v, _ = np.linalg.eig(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1. / (v + k))


def get_batch_jacobian(net, x, target):
    '''
    Returns gradient of net prediction on x and detached target
    '''
    net.eval()
    net.zero_grad()
    x.requires_grad_(True)
    # this is prediction
    y = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach()


def calculate_dw(model, inputs, labels, loss_fn):
    model.zero_grad()
    target_loss, _, _ = loss_fn(model(inputs), labels)
    dw = torch.autograd.grad(target_loss, model.parameters())
    return dw


def cal_dis(a, b, metric='L2'):
    '''
    This function calls the metric with flattened a and b.
    Metric should be given as a  string:
    'L2' - L2 norm
    'L1' - L1 norm
    'cos' - cosine similarity
    '''
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
    '''
    Function used for getting images and labels from dataset
    '''
    ground_truth, labels = [], []
    for idx in idx_list:
        img, label = loader.dataset[idx]
        if label not in labels:
            labels.append(torch.as_tensor((label,), device=setup['device']))
            ground_truth.append(img.to(**setup))
        if len(labels) >= max_num:
            break
    ground_truth = torch.stack(ground_truth)
    labels = torch.cat(labels)
    return ground_truth, labels


def accuracy_metric(idx_list, model, loss_fn, trainloader, validloader, setup):
    '''
    Implementation of 4.3 part of the paper.
    '''

    ground_truth, labels = prepare_data(idx_list, validloader, setup)
    model.zero_grad()
    jacobs, labels = get_batch_jacobian(model, ground_truth, labels)
    jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
    return eval_score(jacobs)


def reconstruct(idx, model, loss_fn, trainloader, validloader, setup, opt):
    '''
    Implementation of part 4.2 of the paper
    '''
    if opt.data == 'cifar100':
        dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    elif opt.data == 'FashionMinist':
        dm = torch.Tensor([0.1307]).view(1, 1, 1).to(setup['device'])
        ds = torch.Tensor([0.3081]).view(1, 1, 1).to(setup['device'])
    else:
        raise NotImplementedError

    ground_truth, labels = prepare_data(list(range(idx, idx+len(validloader))), validloader, setup, opt.num_images)
    model.zero_grad()

    # calcuate ori dW
    target_loss, _, _ = loss_fn(model(ground_truth), labels)
    input_gradient = torch.autograd.grad(target_loss, model.parameters())

    # attack model
    model.eval()
    dw_list = list()
    bin_num = 20
    noise_input = (torch.rand((ground_truth.shape)).to(setup['device']) - dm) / ds
    for dis_iter in range(bin_num + 1):
        model.zero_grad()
        fake_ground_truth = (1.0 / bin_num * dis_iter * ground_truth + 1. / bin_num * (bin_num - dis_iter) * noise_input).detach()
        fake_dw = calculate_dw(model, fake_ground_truth, labels, loss_fn)
        dw_loss = sum([cal_dis(dw_a, dw_b, metric='cos') for dw_a, dw_b in zip(fake_dw, input_gradient)]) / len(input_gradient)
        dw_list.append(dw_loss)

    interval_distance = cal_dis(noise_input, ground_truth, metric='L1') / bin_num

    return area_ratio(dw_list, interval_distance, bin_num)


def area_ratio(y_list, inter, bin_num):
    area = 0
    max_area = inter * bin_num
    for idx in range(1, len(y_list)):
        prev = y_list[idx - 1]
        cur = y_list[idx]
        area += (prev + cur) * inter / 2
    return area / max_area


def main(opt):
    # init env
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative')
    defs.epochs = opt.epochs

    loss_fn, trainloader, validloader = preprocess(opt, defs, valid=True)
    model = create_model(opt)
    model.to(**setup)
    old_state_dict = copy.deepcopy(model.state_dict())
    model.load_state_dict(torch.load('checkpoints/tiny_data_{}_arch_{}/{}.pth'.format(opt.data, opt.arch, opt.epochs), map_location=setup["device"]))

    model.eval()

    # at this step model should be loaded from checkpoint and set to eval
    metric_list = list()
    start = time.time()
    # sample indices of the data???
    sample_list = [200 + i * 5 for i in range(100)]

    #  for each sample(?) get metrics from reconstruction
    for attack_id, idx in enumerate(sample_list):
        metric = reconstruct(idx, model, loss_fn, trainloader, validloader, setup, opt)
        metric_list.append(metric)
        print('attach {}th in {}, metric {}'.format(attack_id, opt.aug_list, metric))

    pathname = 'search/data_{}_arch_{}/{}'.format(opt.data, opt.arch, opt.aug_list)
    root_dir = os.path.dirname(pathname)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if len(metric_list) > 0:
        print("Mean of metric list:", np.mean(metric_list))
        np.save(pathname, metric_list)

    # maybe need old_state_dict
    model.load_state_dict(old_state_dict)
    score_list = list()
    for run in range(10):
        large_samle_list = [200 + run * 100 + i for i in range(100)]
        score = accuracy_metric(large_samle_list, model, loss_fn, trainloader, validloader, setup, opt)
        score_list.append(score)

    print('time cost ', time.time() - start)

    pathname = 'accuracy/data_{}_arch_{}/{}'.format(opt.data, opt.arch, opt.aug_list)
    root_dir = os.path.dirname(pathname)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    np.save(pathname, score_list)
    print(score_list)
