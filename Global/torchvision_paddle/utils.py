from typing import Union, Optional, List, Tuple, Text, BinaryIO
import pathlib
import math
import paddle
import warnings
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor

__all__ = ["make_grid", "save_image", "draw_bounding_boxes"]


@paddle.no_grad()
def make_grid(
        tensor: Union[paddle.Tensor, List[paddle.Tensor]],
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        value_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
        **kwargs
) -> paddle.Tensor:
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (paddle.is_tensor(tensor) or
            (isinstance(tensor, list) and all(paddle.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if "range" in kwargs.keys():
        warning = "range will be deprecated, please use value_range instead."
        warnings.warn(warning)
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = paddle.stack(tensor, axis=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = paddle.concat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = paddle.concat((tensor, tensor, tensor), 1)

    if normalize is True:
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img = img / max(high - low, 1e-5)

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(paddle.min(t)), float(paddle.max(t)))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] +
                                                        padding)
    num_channels = tensor.shape[1]
    grid = paddle.full((num_channels, height * ymaps + padding,
                        width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[:, y * height + padding:(y + 1) * height, x * width + padding:(x + 1) * width] = tensor[k]
            k = k + 1
    return grid


@paddle.no_grad()
def save_image(
        tensor: Union[paddle.Tensor, List[paddle.Tensor]],
        fp: Union[Text, pathlib.Path, BinaryIO],
        format: Optional[str] = None,
        **kwargs
) -> None:
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = make_grid(tensor, **kwargs)
    ndarr = paddle.clip(grid * 255 + 0.5, 0, 255).transpose(
        [1, 2, 0]).cast("uint8").numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)


@paddle.no_grad()
def draw_bounding_boxes(
        image: paddle.Tensor,
        boxes: paddle.Tensor,
        labels: Optional[List[str]] = None,
        colors: Optional[List[Union[str, Tuple[int, int, int]]]] = None,
        fill: Optional[bool] = False,
        width: int = 1,
        font: Optional[str] = None,
        font_size: int = 10
) -> paddle.Tensor:
    """
    Draws bounding boxes on given image.
    The values of the input image should be uint8 between 0 and 255.
    If filled, Resulting Tensor should be saved as PNG image.

    Args:
        image (Tensor): Tensor of shape (C x H x W)
        boxes (Tensor): Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format. Note that
            the boxes are absolute coordinates with respect to the image. In other words: `0 <= xmin < xmax < W` and
            `0 <= ymin < ymax < H`.
        labels (List[str]): List containing the labels of bounding boxes.
        colors (List[Union[str, Tuple[int, int, int]]]): List containing the colors of bounding boxes. The colors can
            be represented as `str` or `Tuple[int, int, int]`.
        fill (bool): If `True` fills the bounding box with specified color.
        width (int): Width of bounding box.
        font (str): A filename containing a TrueType font. If the file is not found in this filename, the loader may
            also search in other directories, such as the `fonts/` directory on Windows or `/Library/Fonts/`,
            `/System/Library/Fonts/` and `~/Library/Fonts/` on macOS.
        font_size (int): The requested font size in points.
    """

    if not isinstance(image, paddle.Tensor):
        raise TypeError(f"Tensor expected, got {type(image)}")
    elif image.dtype != paddle.uint8:
        raise ValueError(f"Tensor uint8 expected, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")

    ndarr = image.permute(1, 2, 0).numpy()
    img_to_draw = Image.fromarray(ndarr)

    img_boxes = boxes.to(paddle.int64).tolist()

    if fill:
        draw = ImageDraw.Draw(img_to_draw, "RGBA")

    else:
        draw = ImageDraw.Draw(img_to_draw)

    txt_font = ImageFont.load_default() if font is None else ImageFont.truetype(font=font, size=font_size)

    for i, bbox in enumerate(img_boxes):
        if colors is None:
            color = None
        else:
            color = colors[i]

        if fill:
            if color is None:
                fill_color = (255, 255, 255, 100)
            elif isinstance(color, str):
                # This will automatically raise Error if rgb cannot be parsed.
                fill_color = ImageColor.getrgb(color) + (100,)
            elif isinstance(color, tuple):
                fill_color = color + (100,)
            else:
                fill_color = (255, 255, 255, 100)
            draw.rectangle(bbox, width=width, outline=color, fill=fill_color)
        else:
            draw.rectangle(bbox, width=width, outline=color)

        if labels is not None:
            draw.text((bbox[0], bbox[1]), labels[i], fill=color, font=txt_font)

    return paddle.to_tensor(np.array(img_to_draw)).transpose([2, 0, 1])
