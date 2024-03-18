import random
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np


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
