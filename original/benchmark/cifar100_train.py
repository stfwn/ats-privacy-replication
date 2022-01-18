import os, sys
import torch
import random
import original.inversefed as inversefed
import argparse
import original.policy as policy
from original.benchmark.comm import create_model, preprocess

sys.path.insert(0, "./")
seed = 23333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
policies = policy.policies

parser = argparse.ArgumentParser(
    description="Reconstruct some image from a trained model."
)
parser.add_argument(
    "--arch", default=None, required=True, type=str, help="Vision model."
)
parser.add_argument(
    "--data", default=None, required=True, type=str, help="Vision dataset."
)
parser.add_argument(
    "--epochs", default=None, required=True, type=int, help="Vision epoch."
)
parser.add_argument(
    "--aug_list", default=None, required=True, type=str, help="Augmentation method."
)
parser.add_argument("--mode", default=None, required=True, type=str, help="Mode.")
parser.add_argument("--rlabel", default=False, type=bool, help="remove label.")
parser.add_argument("--evaluate", default=False, type=bool, help="Evaluate")

opt = parser.parse_args()

# init env
setup = inversefed.utils.system_startup()
defs = inversefed.training_strategy("conservative")
defs.epochs = opt.epochs

# init training
arch = opt.arch
trained_model = True
mode = opt.mode
assert mode in ["normal", "aug", "crop"]


def create_save_dir():
    return "checkpoints/data_{}_arch_{}_mode_{}_auglist_{}_rlabel_{}".format(
        opt.data, opt.arch, opt.mode, opt.aug_list, opt.rlabel
    )


def main():
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy("conservative")
    defs.epochs = (
        opt.epochs
    )  # STFWN: They needed this fix because there is a typo in the conversative training strategy
    loss_fn, trainloader, validloader = preprocess(opt, defs)

    # init model
    model = create_model(opt)
    model.to(**setup)
    save_dir = create_save_dir()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file = f"{save_dir}/{arch}_{defs.epochs}.pth"
    inversefed.train(
        model, loss_fn, trainloader, validloader, defs, setup=setup, save_dir=save_dir
    )
    torch.save(model.state_dict(), f"{file}")
    model.eval()


def evaluate():
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy("conservative")
    defs.epochs = opt.epochs
    loss_fn, trainloader, validloader = preprocess(opt, defs, valid=False)
    model = create_model(opt)
    model.to(**setup)
    root = create_save_dir()

    filename = os.path.join(root, "{}_{}.pth".format(opt.arch, opt.epochs))
    print(filename)
    if not os.path.exists(filename):
        assert False

    print(filename)
    model.load_state_dict(torch.load(filename))
    model.eval()
    stats = {"valid_losses": list(), "valid_Accuracy": list()}
    inversefed.training.training_routine.validate(
        model, loss_fn, validloader, defs, setup=setup, stats=stats
    )
    print(stats)


if __name__ == "__main__":
    if opt.evaluate:
        evaluate()
        exit(0)
    main()
