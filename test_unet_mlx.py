# Copied from https://github.com/Mostafa-wael/U-Net-in-PyTorch/blob/main/test.py
import copy
import itertools
import random
import time
from collections import defaultdict

import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data_utils import generate_random_data


class SimDataset(Dataset):
    def __init__(self, count, transform=None):
        self.input_images, self.target_masks = generate_random_data(
            192, 192, count=count
        )
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)

        return [image, mask]


def get_data_loaders():
    # use the same transformations for train/val in this example
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # imagenet
        ]
    )

    train_set = SimDataset(500, transform=trans)
    val_set = SimDataset(20, transform=trans)

    image_datasets = {"train": train_set, "val": val_set}

    batch_size = 20

    dataloaders = {
        "train": DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=0
        ),
        "val": DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0),
    }

    return dataloaders


def dice_loss(pred, target, smooth=1.0):
    intersection = (pred * target).sum(axis=2).sum(axis=2)

    loss = 1 - (
        (2.0 * intersection + smooth)
        / (pred.sum(axis=2).sum(axis=2) + target.sum(axis=2).sum(axis=2) + smooth)
    )
    return loss.mean()


def loss_fn(model, inputs, target, metrics, bce_weight=0.5):
    pred = model(inputs)
    bce = nn.losses.binary_cross_entropy(pred, target, with_logits=True)

    pred = mx.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics["bce"] += np.array(bce, copy=False) * target.shape[0]
    metrics["dice"] += np.array(dice, copy=False) * target.shape[0]
    metrics["loss"] += np.array(loss, copy=False) * target.shape[0]

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, optimizer, num_epochs=25, learning_rate=3e-4, lr_warmup=200):
    dataloaders = get_data_loaders()
    model.save_weights("best_weights.npz")
    best_loss = 1e10

    iterations = 0
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = mx.array(inputs.numpy().transpose([0, 2, 3, 1]))
                labels = mx.array(labels.numpy().transpose([0, 2, 3, 1]))
                # forward
                loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
                loss, grads = loss_and_grad_fn(model, inputs, labels, metrics)

                # backward + optimize only if in training phase
                if phase == "train":
                    optimizer.update(model, grads)

                # statistics
                epoch_samples += inputs.shape[0]

            if phase == "train":
                # Manual learning rate scheduler
                iterations += 1
                optimizer.learning_rate = min(1, iterations / lr_warmup) * learning_rate
                print(f"Learning rate: {optimizer.learning_rate}")

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics["loss"] / epoch_samples

            # deep copy the model
            if phase == "val" and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                model.save_weights("best_weights.npz")

        time_elapsed = time.time() - since
        print("{:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    print("Best val loss: {:4f}".format(best_loss))
    return model


def run(UNet):
    num_class = 6

    model = UNet(num_class)

    optimizer_ft = optim.Adam(learning_rate=1e-4)

    model = train_model(model, optimizer_ft, num_epochs=60)

    model.eval()  # Set model to the evaluation mode

    # load best model weights
    model.load_weights("best_weights.npz")

    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # imagenet
        ]
    )
    # # Create another simulation dataset for test
    test_dataset = SimDataset(3, transform=trans)
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=0)

    # Get the first batch
    inputs, labels = next(iter(test_loader))

    # Predict
    pred = model(mx.array(inputs.numpy().transpose([0, 2, 3, 1])))
    # The loss functions include the sigmoid function.
    pred = mx.sigmoid(pred)

    pred = np.array(pred.transpose([0, 3, 1, 2]), copy=False)

    # Change channel-order and make 3 channels for matplot
    input_images_rgb = [reverse_transform(x) for x in inputs]

    # Map each channel (i.e. class) to each color
    target_masks_rgb = [masks_to_colorimg(x) for x in labels]
    pred_rgb = [masks_to_colorimg(x) for x in pred]

    plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb])
