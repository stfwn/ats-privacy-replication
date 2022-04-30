import torch

from pytorch_lightning import LightningModule

from thirdparty import inversefed
from data_modules import DataModule


def attack(
    model: LightningModule,
    data_module: DataModule,
    optimizer: str,
    img_idx: int,
    rlabel: bool = False,
    max_iterations: int = None,
):
    """Wraps inversefed.GradientReconstructor to reconstruct one image at a time."""

    # Init data
    data_module.prepare_data()
    data_module.setup(stage="test")
    img, label = data_module.test[img_idx]
    img = img.view(1, *img.shape).to(model.device)
    label = torch.tensor(label).view(1).to(model.device)
    data_mean = data_module.mean.view(data_module.num_channels, 1, 1).to(
        model.device
    )
    data_std = data_module.std.view(data_module.num_channels, 1, 1).to(
        model.device
    )

    # Compute gradient
    model.eval()
    model.zero_grad()
    model.loss_function.factor = 1
    loss = model.loss_function(model(img), label)
    params = [p for p in model.parameters() if p.requires_grad]
    gradient = torch.autograd.grad(loss, params)

    # Reconstruct
    config = create_config(optimizer)
    if max_iterations:
        config["max_iterations"] = max_iterations
    rec_machine = inversefed.GradientReconstructor(
        model, (data_mean, data_std), config, num_images=1
    )
    # Omitting the reconstruction stats so it can be runable on lisa
    output, reconstruction_stats = rec_machine.reconstruct(
        gradient,
        None if rlabel else label,
        img_shape=data_module.dims,
    )

    # Denormalize
    output_denormalized = output * data_std + data_mean
    input_denormalized = img * data_std + data_mean

    # Compute metrics
    img_mse = (output_denormalized - input_denormalized).pow(2).mean()
    pred_mse = (model(output) - model(img)).pow(2).mean()
    psnr = inversefed.metrics.psnr(output_denormalized, input_denormalized)

    return (
        input_denormalized,
        output_denormalized,
        reconstruction_stats,
        psnr,
        img_mse,
        pred_mse,
    )


# original code to get a configuration dict for RecMachine (maybe turn this into something prettier)
def create_config(optimizer):
    if optimizer == "inversed":
        # AKA 'adam-cosine'
        config = dict(
            signed=True,
            boxed=True,
            cost_fn="sim",
            indices="def",
            weights="equal",
            lr=0.1,
            optim="adam",
            restarts=1,
            max_iterations=4800,
            total_variation=1e-4,
            init="randn",
            filter="none",
            lr_decay=True,
            scoring_choice="loss",
        )
    elif optimizer == "inversed-zero":
        config = dict(
            signed=True,
            boxed=True,
            cost_fn="sim",
            indices="def",
            weights="equal",
            lr=0.1,
            optim="adam",
            restarts=1,
            max_iterations=4800,
            total_variation=1e-4,
            init="zeros",
            filter="none",
            lr_decay=True,
            scoring_choice="loss",
        )
    elif optimizer == "inversed-sim-out":
        config = dict(
            signed=True,
            boxed=True,
            cost_fn="out_sim",
            indices="def",
            weights="equal",
            lr=0.1,
            optim="adam",
            restarts=1,
            max_iterations=4800,
            total_variation=1e-4,
            init="zeros",
            filter="none",
            lr_decay=True,
            scoring_choice="loss",
        )
    elif optimizer == "inversed-sgd-sim":
        config = dict(
            signed=True,
            boxed=True,
            cost_fn="sim",
            indices="def",
            weights="equal",
            lr=0.1,
            optim="sgd",
            restarts=1,
            max_iterations=4800,
            total_variation=1e-4,
            init="randn",
            filter="none",
            lr_decay=True,
            scoring_choice="loss",
        )
    elif optimizer == "inversed-LBFGS-sim":
        config = dict(
            signed=True,
            boxed=True,
            cost_fn="sim",
            indices="def",
            weights="equal",
            lr=1e-4,
            optim="LBFGS",
            restarts=16,
            max_iterations=300,
            total_variation=1e-4,
            init="randn",
            filter="none",
            lr_decay=False,
            scoring_choice="loss",
        )
    elif optimizer == "inversed-adam-L1":
        config = dict(
            signed=True,
            boxed=True,
            cost_fn="l1",
            indices="def",
            weights="equal",
            lr=0.1,
            optim="adam",
            restarts=1,
            max_iterations=4800,
            total_variation=1e-4,
            init="randn",
            filter="none",
            lr_decay=True,
            scoring_choice="loss",
        )
    elif optimizer == "inversed-adam-L2":
        config = dict(
            signed=True,
            boxed=True,
            cost_fn="l2",
            indices="def",
            weights="equal",
            lr=0.1,
            optim="adam",
            restarts=1,
            max_iterations=4800,
            total_variation=1e-4,
            init="randn",
            filter="none",
            lr_decay=True,
            scoring_choice="loss",
        )
    elif optimizer == "zhu":
        config = dict(
            signed=False,
            boxed=False,
            cost_fn="l2",
            indices="def",
            weights="equal",
            lr=1e-4,
            optim="LBFGS",
            restarts=2,
            max_iterations=50,  # ??
            total_variation=1e-3,
            init="randn",
            filter="none",
            lr_decay=False,
            scoring_choice="loss",
        )
    elif optimizer == "inversefed-default":
        # InverseFed default, not included in cifar100_attack.py
        config = dict(
            signed=False,
            boxed=True,
            cost_fn="sim",
            indices="def",
            weights="equal",
            lr=0.1,
            optim="adam",
            restarts=1,
            max_iterations=4800,
            total_variation=1e-1,
            init="randn",
            filter="none",
            lr_decay=True,
            scoring_choice="loss",
        )
    else:
        raise ValueError
    return config
