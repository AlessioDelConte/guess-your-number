from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Accuracy, Loss, RunningAverage
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, Grayscale

from draw_image import draw_image


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def get_data_loaders(train_batch_size, val_batch_size):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(
            MNIST(download=True, root=".", transform=data_transform, train=True), batch_size=train_batch_size,
            shuffle=True
    )

    val_loader = DataLoader(
            MNIST(download=False, root=".", transform=data_transform, train=False), batch_size=val_batch_size,
            shuffle=False
    )
    return train_loader, val_loader


def run(train_batch_size, val_batch_size, epochs, lr, momentum, display_gpu_info, eval):
    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
    model = Net()
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    trainer = create_supervised_trainer(model, optimizer, F.nll_loss, device=device)
    evaluator = create_supervised_evaluator(model, metrics={"accuracy": Accuracy(), "nll": Loss(F.nll_loss)},
                                            device=device)

    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

    if display_gpu_info:
        from ignite.contrib.metrics import GpuInfo

        GpuInfo().attach(trainer, name="gpu")

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names="all")

    def score_function(engine):
        val_loss = engine.state.metrics['nll']
        return -val_loss

    stopping_handler = EarlyStopping(patience=3, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, stopping_handler)

    saving_handler = ModelCheckpoint('models', 'MNIST', n_saved=2, create_dir=True, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=2), saving_handler, {'mymodel': model})

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        pbar.log_message(
                "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                        engine.state.epoch, avg_accuracy, avg_nll
                )
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        pbar.log_message(
                "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                        engine.state.epoch, avg_accuracy, avg_nll
                )
        )

        pbar.n = pbar.last_print_n = 0

    if eval:
        model.load_state_dict(torch.load("models/MNIST_model.pth"))
        draw_image()
        image = Image.open("number.png")
        image = ImageOps.invert(image)

        class OneHotNormalization(object):
            def __call__(self, tensor):
                for i in range(tensor.shape[1]):
                    for j in range(tensor.shape[2]):
                        if tensor[0, i, j] > 0:
                            tensor[0, i, j] += .45
                return tensor

        image = Compose([Grayscale(), Resize((28, 28), interpolation=5), ToTensor(), OneHotNormalization(),
                         Normalize((0.1307,), (0.3081,))])(image)
        image = image.reshape(28, 28)
        plt.imshow(image)
        plt.show()
        model.eval()
        image = image.reshape(1, 1, 28, 28)
        print("Hai disegnato un: ", torch.argmax(model(image)).item())
    else:
        trainer.run(train_loader, max_epochs=epochs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="input batch size for training (default: 64)")
    parser.add_argument(
            "--val_batch_size", type=int, default=1000, help="input batch size for validation (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument(
            "--display_gpu_info",
            action="store_true",
            help="Display gpu usage info. This needs python 3.X and pynvml package",
    )
    parser.add_argument("--eval", type=bool, default=True, help="Eval mode")

    args = parser.parse_args()

    run(args.batch_size, args.val_batch_size, args.epochs, args.lr, args.momentum, args.display_gpu_info, args.eval)
