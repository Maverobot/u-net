# Copied from https://github.com/Mostafa-wael/U-Net-in-PyTorch/blob/main/test.py
import copy
import itertools
import random
import time
from collections import defaultdict
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


def generate_random_data(height, width, count):
    x, y = zip(*[generate_img_and_mask(height, width) for i in range(0, count)])

    X = np.asarray(x) * 255
    X = X.repeat(3, axis=1).transpose([0, 2, 3, 1]).astype(np.uint8)
    Y = np.asarray(y)

    return X, Y


def generate_img_and_mask(height, width):
    shape = (height, width)

    triangle_location = get_random_location(*shape)
    circle_location1 = get_random_location(*shape, zoom=0.7)
    circle_location2 = get_random_location(*shape, zoom=0.5)
    mesh_location = get_random_location(*shape)
    square_location = get_random_location(*shape, zoom=0.8)
    plus_location = get_random_location(*shape, zoom=1.2)

    # Create input image
    arr = np.zeros(shape, dtype=bool)
    arr = add_triangle(arr, *triangle_location)
    arr = add_circle(arr, *circle_location1)
    arr = add_circle(arr, *circle_location2, fill=True)
    arr = add_mesh_square(arr, *mesh_location)
    arr = add_filled_square(arr, *square_location)
    arr = add_plus(arr, *plus_location)
    arr = np.reshape(arr, (1, height, width)).astype(np.float32)

    # Create target masks
    masks = np.asarray(
        [
            add_filled_square(np.zeros(shape, dtype=bool), *square_location),
            add_circle(np.zeros(shape, dtype=bool), *circle_location2, fill=True),
            add_triangle(np.zeros(shape, dtype=bool), *triangle_location),
            add_circle(np.zeros(shape, dtype=bool), *circle_location1),
            add_filled_square(np.zeros(shape, dtype=bool), *mesh_location),
            # add_mesh_square(np.zeros(shape, dtype=bool), *mesh_location),
            add_plus(np.zeros(shape, dtype=bool), *plus_location),
        ]
    ).astype(np.float32)

    return arr, masks


def add_square(arr, x, y, size):
    s = int(size / 2)
    arr[x - s, y - s : y + s] = True
    arr[x + s, y - s : y + s] = True
    arr[x - s : x + s, y - s] = True
    arr[x - s : x + s, y + s] = True

    return arr


def add_filled_square(arr, x, y, size):
    s = int(size / 2)

    xx, yy = np.mgrid[: arr.shape[0], : arr.shape[1]]

    return np.logical_or(
        arr, logical_and([xx > x - s, xx < x + s, yy > y - s, yy < y + s])
    )


def logical_and(arrays):
    new_array = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        new_array = np.logical_and(new_array, a)

    return new_array


def add_mesh_square(arr, x, y, size):
    s = int(size / 2)

    xx, yy = np.mgrid[: arr.shape[0], : arr.shape[1]]

    return np.logical_or(
        arr,
        logical_and(
            [xx > x - s, xx < x + s, xx % 2 == 1, yy > y - s, yy < y + s, yy % 2 == 1]
        ),
    )


def add_triangle(arr, x, y, size):
    s = int(size / 2)

    triangle = np.tril(np.ones((size, size), dtype=bool))

    arr[x - s : x - s + triangle.shape[0], y - s : y - s + triangle.shape[1]] = triangle

    return arr


def add_circle(arr, x, y, size, fill=False):
    xx, yy = np.mgrid[: arr.shape[0], : arr.shape[1]]
    circle = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    new_arr = np.logical_or(
        arr, np.logical_and(circle < size, circle >= size * 0.7 if not fill else True)
    )

    return new_arr


def add_plus(arr, x, y, size):
    s = int(size / 2)
    arr[x - 1 : x + 1, y - s : y + s] = True
    arr[x - s : x + s, y - 1 : y + 1] = True

    return arr


def get_random_location(width, height, zoom=1.0):
    x = int(width * random.uniform(0.1, 0.9))
    y = int(height * random.uniform(0.1, 0.9))

    size = int(min(width, height) * random.uniform(0.06, 0.12) * zoom)

    return (x, y, size)


def plot_img_array(img_array, ncol=3):
    nrow = len(img_array) // ncol

    f, plots = plt.subplots(
        nrow, ncol, sharex="all", sharey="all", figsize=(ncol * 4, nrow * 4)
    )

    for i in range(len(img_array)):
        plots[i // ncol, i % ncol]
        plots[i // ncol, i % ncol].imshow(img_array[i])
    plt.show()


def plot_side_by_side(img_arrays):
    flatten_list = reduce(lambda x, y: x + y, zip(*img_arrays))

    plot_img_array(np.array(flatten_list), ncol=len(img_arrays))


def plot_errors(results_dict, title):
    markers = itertools.cycle(("+", "x", "o"))

    plt.title("{}".format(title))

    for label, result in sorted(results_dict.items()):
        plt.plot(result, marker=next(markers), label=label)
        plt.ylabel("dice_coef")
        plt.xlabel("epoch")
        plt.legend(loc=3, bbox_to_anchor=(1, 0))

    plt.show()


def masks_to_colorimg(masks):
    colors = np.asarray(
        [
            (201, 58, 64),
            (242, 207, 1),
            (0, 152, 75),
            (101, 172, 228),
            (56, 34, 132),
            (160, 194, 56),
        ]
    )

    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:, y, x] > 0.5]

            if len(selected_colors) > 0:
                colorimg[y, x, :] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)


def generate_images_and_masks_then_plot():
    # Generate some random images
    input_images, target_masks = generate_random_data(192, 192, count=3)

    for x in [input_images, target_masks]:
        print(x.shape)
        print(x.min(), x.max())

    # Change channel-order and make 3 channels for matplot
    input_images_rgb = [x.astype(np.uint8) for x in input_images]

    # Map each channel (i.e. class) to each color
    target_masks_rgb = [masks_to_colorimg(x) for x in target_masks]

    # Left: Input image (black and white), Right: Target mask (6ch)
    plot_side_by_side([input_images_rgb, target_masks_rgb])


def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp


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
