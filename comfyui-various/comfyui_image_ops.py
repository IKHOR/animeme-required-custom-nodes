import json
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from torchvision.transforms import InterpolationMode

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def register_node(identifier: str, display_name: str):
    def decorator(cls):
        NODE_CLASS_MAPPINGS[identifier] = cls
        NODE_DISPLAY_NAME_MAPPINGS[identifier] = display_name

        return cls

    return decorator


def load_image(path, convert="RGB"):
    img = Image.open(path).convert(convert)
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).unsqueeze(0)
    return img


def save_image(img: torch.Tensor, path, prompt=None, extra_pnginfo: dict = None):
    path = str(path)

    if len(img.shape) != 3:
        raise ValueError(f"can't take image batch as input, got {img.shape[0]} images")

    img = img.permute(2, 0, 1)
    if img.shape[0] != 3:
        raise ValueError(f"image must have 3 channels, but got {img.shape[0]} channels")

    img = img.clamp(0, 1)
    img = F.to_pil_image(img)

    metadata = PngInfo()

    if prompt is not None:
        metadata.add_text("prompt", json.dumps(prompt))

    if extra_pnginfo is not None:
        for k, v in extra_pnginfo.items():
            metadata.add_text(k, json.dumps(v))

    img.save(path, pnginfo=metadata, compress_level=4)

    subfolder, filename = os.path.split(path)

    return {"filename": filename, "subfolder": subfolder, "type": "output"}


@register_node("JWImageLoadRGB", "Image Load RGB")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "path": ("STRING", {"default": "./image.png"}),
        }
    }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(self, path: str):
        assert isinstance(path, str)

        img = load_image(path)
        return (img,)


@register_node("JWImageLoadRGBA", "Image Load RGBA")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "path": ("STRING", {"default": "./image.png"}),
        }
    }
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "execute"

    def execute(self, path: str):
        assert isinstance(path, str)

        img = load_image(path, convert="RGBA")
        color = img[:, :, :, 0:3]
        mask = img[0, :, :, 3]
        mask = 1 - mask  # invert mask

        return (color, mask)


@register_node("JWLoadImagesFromString", "Load Images From String")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "paths": (
                "STRING",
                {
                    "default": "./frame000001.png\n./frame000002.png\n./frame000003.png",
                    "multiline": True,
                    "dynamicPrompts": False,
                },
            ),
            "ignore_missing_images": (("false", "true"), {"default": "false"}),
        }
    }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(self, paths, ignore_missing_images: str):
        assert isinstance(paths, str)
        assert isinstance(ignore_missing_images, str)

        ignore_missing_images: bool = ignore_missing_images == "true"

        paths = [p.strip() for p in paths.splitlines()]
        paths = [p for p in paths if len(p) != 0]

        if ignore_missing_images:
            # remove missing images
            paths = [p for p in paths if os.path.exists(p)]
        else:
            # early check for missing images
            for path in paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Image does not exist: {path}")

        if len(paths) == 0:
            raise RuntimeError("Image sequence empty - no images to load")

        imgs = []
        for path in paths:
            img = load_image(path)
            # img.shape => torch.Size([1, 768, 768, 3])
            imgs.append(img)

        imgs = torch.cat(imgs, dim=0)

        return (imgs,)


@register_node("JWImageSaveToPath", "Image Save To Path")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "path": ("STRING", {"default": "./image.png"}),
            "image": ("IMAGE",),
            "overwrite": (("false", "true"), {"default": "true"}),
        },
        "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
    }
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "execute"

    def execute(
        self,
        path: str,
        image: torch.Tensor,
        overwrite: str,
        prompt=None,
        extra_pnginfo=None,
    ):
        assert isinstance(path, str)
        assert isinstance(image, torch.Tensor)
        assert isinstance(overwrite, str)

        overwrite: bool = overwrite == "true"

        path: Path = Path(path)
        if not overwrite and path.exists():
            return ()

        path.parent.mkdir(exist_ok=True)

        if image.shape[0] == 1:
            # batch has 1 image only
            save_image(
                image[0],
                path,
                prompt=prompt,
                extra_pnginfo=extra_pnginfo,
            )
        else:
            # batch has multiple images
            for i, img in enumerate(image):
                subpath = path.with_stem(f"{path.stem}-{i}")
                save_image(
                    img,
                    subpath,
                    prompt=prompt,
                    extra_pnginfo=extra_pnginfo,
                )

        return ()


@register_node("JWImageExtractFromBatch", "Image Extract From Batch")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "images": ("IMAGE",),
            "index": ("INT", {"default": 0, "min": 0, "step": 1}),
        }
    }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(self, images: torch.Tensor, index: int):
        assert isinstance(images, torch.Tensor)
        assert isinstance(index, int)

        img = images[index].unsqueeze(0)

        return (img,)


@register_node("JWImageBatchCount", "Get Image Batch Count")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "images": ("IMAGE",),
        }
    }
    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, images: torch.Tensor):
        assert isinstance(images, torch.Tensor)

        batch_count = len(images)

        return (batch_count,)


@register_node("JWImageResize", "Image Resize")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "image": ("IMAGE",),
            "height": ("INT", {"default": 512, "min": 0, "step": 1, "max": 99999}),
            "width": ("INT", {"default": 512, "min": 0, "step": 1, "max": 99999}),
            "interpolation_mode": (
                ["bicubic", "bilinear", "nearest", "nearest exact"],
            ),
        }
    }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(
        self,
        image: torch.Tensor,
        width: int,
        height: int,
        interpolation_mode: str,
    ):
        assert isinstance(image, torch.Tensor)
        assert isinstance(height, int)
        assert isinstance(width, int)
        assert isinstance(interpolation_mode, str)

        interpolation_mode = interpolation_mode.upper().replace(" ", "_")
        interpolation_mode = getattr(InterpolationMode, interpolation_mode)

        image = image.permute(0, 3, 1, 2)
        image = F.resize(
            image,
            (height, width),
            interpolation=interpolation_mode,
            antialias=True,
        )
        image = image.permute(0, 2, 3, 1)

        return (image,)


@register_node("JWImageFlip", "Image Flip")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "image": ("IMAGE",),
            "direction": (("horizontal", "vertical"), {"default": "hotizontal"}),
        }
    }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(
        self,
        image: torch.Tensor,
        direction: str,
    ):
        assert isinstance(image, torch.Tensor)
        assert direction in ("horizontal", "vertical")

        image = image.permute(0, 3, 1, 2)
        if direction == "horizontal":
            image = F.hflip(image)
        else:
            image = F.vflip(image)
        image = image.permute(0, 2, 3, 1)

        return (image,)


@register_node("JWMaskResize", "Mask Resize")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "mask": ("MASK",),
            "height": ("INT", {"default": 512, "min": 0, "step": 1, "max": 99999}),
            "width": ("INT", {"default": 512, "min": 0, "step": 1, "max": 99999}),
            "interpolation_mode": (
                ["bicubic", "bilinear", "nearest", "nearest exact"],
            ),
        }
    }
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"

    def execute(
        self,
        mask: torch.Tensor,
        width: int,
        height: int,
        interpolation_mode: str,
    ):
        assert isinstance(mask, torch.Tensor)
        assert isinstance(height, int)
        assert isinstance(width, int)
        assert isinstance(interpolation_mode, str)

        interpolation_mode = interpolation_mode.upper().replace(" ", "_")
        interpolation_mode = getattr(InterpolationMode, interpolation_mode)

        mask = mask.unsqueeze(0)
        image = F.resize(
            image,
            (height, width),
            interpolation=interpolation_mode,
            antialias=True,
        )
        mask = mask[0]

        return (mask,)


@register_node("JWMaskLikeImageSize", "Mask Like Image Size")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "image": ("IMAGE",),
            "value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        }
    }
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"

    def execute(
        self,
        image: torch.Tensor,
        value: float,
    ):
        assert isinstance(image, torch.Tensor)
        assert isinstance(value, float)

        _, h, w, _ = image.shape
        mask_shape = (h, w)
        # code copied from:
        # comfy_extras\nodes_mask.py
        mask = torch.full(mask_shape, value, dtype=torch.float32, device="cpu")

        return (mask,)


@register_node("JWImageResizeToSquare", "Image Resize to Square")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "image": ("IMAGE",),
            "size": ("INT", {"default": 512, "min": 0, "step": 1, "max": 99999}),
            "interpolation_mode": (
                ["bicubic", "bilinear", "nearest", "nearest exact"],
            ),
        }
    }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(
        self,
        image: torch.Tensor,
        size: int,
        interpolation_mode: str,
    ):
        assert isinstance(image, torch.Tensor)
        assert isinstance(size, int)
        assert isinstance(interpolation_mode, str)

        interpolation_mode = interpolation_mode.upper().replace(" ", "_")
        interpolation_mode = getattr(InterpolationMode, interpolation_mode)

        image = image.permute(0, 3, 1, 2)
        image = F.resize(
            image,
            (size, size),
            interpolation=interpolation_mode,
            antialias=True,
        )
        image = image.permute(0, 2, 3, 1)

        return (image,)


@register_node("JWImageResizeByFactor", "Image Resize by Factor")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "image": ("IMAGE",),
            "factor": ("FLOAT", {"default": 1, "min": 0, "step": 0.01, "max": 99999}),
            "interpolation_mode": (
                ["bicubic", "bilinear", "nearest", "nearest exact"],
            ),
        }
    }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(
        self,
        image: torch.Tensor,
        factor: float,
        interpolation_mode: str,
    ):
        assert isinstance(image, torch.Tensor)
        assert isinstance(factor, float)
        assert isinstance(interpolation_mode, str)

        interpolation_mode = interpolation_mode.upper().replace(" ", "_")
        interpolation_mode = getattr(InterpolationMode, interpolation_mode)

        new_height = round(image.shape[1] * factor)
        new_width = round(image.shape[2] * factor)

        image = image.permute(0, 3, 1, 2)
        image = F.resize(
            image,
            (new_height, new_width),
            interpolation=interpolation_mode,
            antialias=True,
        )
        image = image.permute(0, 2, 3, 1)

        return (image,)


@register_node("JWImageResizeByShorterSide", "Image Resize by Shorter Side")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "image": ("IMAGE",),
            "size": ("INT", {"default": 512, "min": 0, "step": 1, "max": 99999}),
            "interpolation_mode": (
                ["bicubic", "bilinear", "nearest", "nearest exact"],
            ),
        }
    }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(
        self,
        image: torch.Tensor,
        size: int,
        interpolation_mode: str,
    ):
        assert isinstance(image, torch.Tensor)
        assert isinstance(size, int)
        assert isinstance(interpolation_mode, str)

        interpolation_mode = interpolation_mode.upper().replace(" ", "_")
        interpolation_mode = getattr(InterpolationMode, interpolation_mode)

        image = image.permute(0, 3, 1, 2)
        image = F.resize(
            image,
            size,
            interpolation=interpolation_mode,
            antialias=True,
        )
        image = image.permute(0, 2, 3, 1)

        return (image,)


@register_node("JWImageResizeByLongerSide", "Image Resize by Longer Side")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "image": ("IMAGE",),
            "size": ("INT", {"default": 512, "min": 0, "step": 1, "max": 99999}),
            "interpolation_mode": (
                ["bicubic", "bilinear", "nearest", "nearest exact"],
            ),
        }
    }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(
        self,
        image: torch.Tensor,
        size: int,
        interpolation_mode: str,
    ):
        assert isinstance(image, torch.Tensor)
        assert isinstance(size, int)
        assert isinstance(interpolation_mode, str)

        interpolation_mode = interpolation_mode.upper().replace(" ", "_")
        interpolation_mode = getattr(InterpolationMode, interpolation_mode)

        _, h, w, _ = image.shape

        if h >= w:
            new_h = size
            new_w = round(w * new_h / h)
        else:  # h < w
            new_w = size
            new_h = round(h * new_w / w)

        image = image.permute(0, 3, 1, 2)
        image = F.resize(
            image,
            (new_w, new_h),
            interpolation=interpolation_mode,
            antialias=True,
        )
        image = image.permute(0, 2, 3, 1)

        return (image,)
