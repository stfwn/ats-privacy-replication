from pytorch_lightning import LightningModule
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock
import torchmetrics


class ModelBase(LightningModule):
    def __init__(self):
        super().__init__()
        # Metrics
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

        self.optimizer = optim.SGD
        self.loss_function = F.cross_entropy

    def configure_optimizers(self):
        optimizer = self.optimizer(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
            nesterov=self.hparams.nesterov,
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                # Decay the learning rate in equal length steps over the course of training
                self.hparams.epochs // 2.667,
                self.hparams.epochs // 1.6,
                self.hparams.epochs // 1.142,
            ],
            gamma=self.hparams.gamma,
        )

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        train_loss = self.loss_function(y_hat, y)
        if self.hparams.bugged_loss:
            train_loss *= 0.5
        self.train_acc(y_hat, y)
        self.log("loss/train", train_loss)
        self.log("acc/train", self.train_acc, on_step=False, on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_function(y_hat, y)
        self.val_acc(y_hat, y)
        self.log("loss/val", val_loss)
        self.log("acc/val", self.val_acc, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.loss_function(y_hat, y)
        self.test_acc(y_hat, y)
        self.log("loss/test", test_loss)
        self.log("acc/test", self.test_acc, on_epoch=True)
        return test_loss

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResNet20(ModelBase):
    """This is a LightningModule version of JonasGeiping's ResNet
    implementation [1], which itself is a copy of torchvision's ResNet [2] with
    added arguments for the number of layers and their strides.

    [1]: https://github.com/JonasGeiping/invertinggradients
    [2]: https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
    """

    _make_layer = ResNet._make_layer

    def __init__(
        self,
        num_channels,
        num_classes,
        epochs,
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        gamma=0.1,
        nesterov=True,
        bugged_loss=True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        # Configure _make_layer :shrug:
        self.base_width = 64
        self.inplanes = 16
        self._norm_layer = nn.BatchNorm2d
        self.dilation = 1
        self.groups = 1

        # Create layers
        width = 16
        self.layers = torch.nn.ModuleList(
            [
                nn.Conv2d(
                    num_channels,
                    self.inplanes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                self._norm_layer(width),
                nn.ReLU(inplace=True),
                self._make_layer(BasicBlock, width, 3, stride=1, dilate=False),
                self._make_layer(
                    BasicBlock, width * 2, 3, stride=2, dilate=False
                ),
                self._make_layer(
                    BasicBlock, width * 4, 3, stride=2, dilate=False
                ),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(
                    (width * 8) // 2 * BasicBlock.expansion,
                    num_classes,
                ),
            ]
        )

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ConvNet(ModelBase):
    def __init__(
        self,
        num_channels,
        num_classes,
        epochs,
        width=16,
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        gamma=0.1,
        nesterov=True,
        bugged_loss=True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(num_channels, width, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(width),
                torch.nn.ReLU(),
                torch.nn.Conv2d(width, 2 * width, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(2 * width),
                torch.nn.ReLU(),
                torch.nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(2 * width),
                torch.nn.ReLU(),
                torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(4 * width),
                torch.nn.ReLU(),
                torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(4 * width),
                torch.nn.ReLU(),
                torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(4 * width),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(3),
                torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(4 * width),
                torch.nn.ReLU(),
                torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(4 * width),
                torch.nn.ReLU(),
                torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(4 * width),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(3),
                torch.nn.Flatten(),
                torch.nn.Linear(36 * width, num_classes),
            ]
        )
