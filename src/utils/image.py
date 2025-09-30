from functools import partial
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.transforms import ToTensor


import numpy as np
import torch

from . import coerce_to_path_and_check_exist, coerce_to_path_and_create_dir, get_files_from_dir
from .logger import print_info, print_warning


IMG_EXTENSIONS = ['jpeg', 'jpg', 'JPG', 'png', 'ppm']


def resize(img, size, keep_aspect_ratio=True, resample=Image.LANCZOS, fit_inside=True):
    if isinstance(size, int):
        return resize(img, (size, size), keep_aspect_ratio=keep_aspect_ratio, resample=resample, fit_inside=fit_inside)
    elif keep_aspect_ratio:
        if fit_inside:
            ratio = float(min([s1 / s2 for s1, s2 in zip(size, img.size)]))  # XXX bug with np.float64 and round
        else:
            ratio = float(max([s1 / s2 for s1, s2 in zip(size, img.size)]))  # XXX bug with np.float64 and round
        size = round(ratio * img.size[0]), round(ratio * img.size[1])
    return img.resize(size, resample=resample)


def adjust_channels(sample, target_channels):
    """
    Adjust the number of channels of a tensor to match n_channels.
    If the tensor has more channels, it averages them.
    If it has fewer, it repeats the channels.
    """
    input_channels = sample.shape[0]
    if input_channels == target_channels:
        return sample
    elif target_channels == 1 and input_channels == 3:
        return sample.mean(0, keepdim=True)  # RGB to grayscale
    elif target_channels == 3 and input_channels == 1:
        return sample.repeat(3, 1, 1)  # Grayscale to RGB
    raise ValueError(f"Cannot convert from {input_channels} to {target_channels} channels")


def unify_channels(x, target_channels):
    if x.shape[1 if x.ndim == 4 else 2] == target_channels:
        return x
    if x.ndim == 4:  # [batch, channels, H, W]
        return torch.stack([adjust_channels(img, target_channels) for img in x])

    # [batch, n_proto, channels, H, W]
    return torch.stack([
        torch.stack([adjust_channels(img, target_channels) for img in batch])
        for batch in x
    ])


def normalize_values(data, target_min=0.1, target_max=0.9, min_range_threshold=0.1):
    """
    Normalize tensor values to a given range [target_min, target_max].
    In order to improve contrast and avoid aberrations
    """
    data = data.detach()

    data_min, data_max = data.min(), data.max()
    data_range = data_max - data_min

    if data_range > min_range_threshold:
        if data_min < 0 or data_max > 1:
            data = (data - data_min) / (data_range + 1e-8)
    else:
        if data_range > 1e-6:
            data = (data - data_min) / (data_range + 1e-8)
            data = data * (target_max - target_min) + target_min
        else:
            # all values are the same, set to mid-range
            data = torch.full_like(data, (target_min + target_max) / 2)

    return data


def gen_checkerboard(h, w, tile_size=8, dark_color=128, light_color=192):
    """Create a checkerboard pattern image of given height and width."""
    checkerboard = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if ((x // tile_size) + (y // tile_size)) % 2 == 0:
                checkerboard[y, x] = light_color
            else:
                checkerboard[y, x] = dark_color
    return checkerboard


def combine_layers(frg=None, mask=None, bkg=None, transparent=False, checkerboard=False):
    """
    Combine foreground, mask, and background into a single image.

    Args:
        frg: Foreground tensor [C, H, W]
        mask: Alpha mask tensor [1, H, W]
        bkg: Background tensor [C, H, W]
        transparent: If True, return RGBA with alpha channel (no background)
        checkerboard: If True, use checkerboard pattern as background

    Returns:
        Combined tensor: [3, H, W] for RGB or [4, H, W] for RGBA
    """
    if frg is not None:
        C, H, W = frg.shape
        device = frg.device
    elif mask is not None:
        _, H, W = mask.shape
        device = mask.device
        C = 3  # Default to RGB
    elif bkg is not None:
        C, H, W = bkg.shape
        device = bkg.device
    else:
        raise ValueError("At least one of frg, mask, or bkg must be provided")

    if frg is None:
        # set to black
        frg = torch.zeros(C, H, W, device=device)

    if mask is None:
        # set to 1 (no masking)
        mask = torch.ones(1, H, W, device=device)

    alpha = mask.expand(C, -1, -1) if mask.shape[0] == 1 else mask
    if transparent:
        return torch.cat([frg, mask], dim=0)  # [4, H, W] = RGB + A

    if checkerboard or bkg is None:
        checker = gen_checkerboard(H, W)
        checker = torch.from_numpy(checker).permute(2, 0, 1).float() / 255.0  # [3, H, W]
        if C == 1:  # Grayscale foreground
            checker = checker.mean(0, keepdim=True)
        bkg = checker.to(device)

    result = bkg * (1 - alpha) + frg * alpha
    return result

def convert_to_img(arr):
    if isinstance(arr, torch.Tensor):
        if len(arr.shape) == 4:
            arr = arr.squeeze(0)
        elif len(arr.shape) == 2:
            arr = arr.unsqueeze(0)
        arr = arr.permute(1, 2, 0).detach().cpu().numpy()

    assert isinstance(arr, np.ndarray)
    if len(arr.shape) == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if np.issubdtype(arr.dtype, np.floating):
        arr = (arr.clip(0, 1) * 255)

    if len(arr.shape) == 3 and arr.shape[2] == 4:
        return Image.fromarray(arr.astype(np.uint8), 'RGBA')
    return Image.fromarray(arr.astype(np.uint8)).convert('RGB')


def convert_to_rgba(t):
    assert isinstance(t, (torch.Tensor,)) and len(t.size()) == 3
    return Image.fromarray((t.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)*255).astype(np.uint8), 'RGBA')


def save_gif(path, name, in_ext='jpg', size=None, total_sec=10):
    files = sorted(get_files_from_dir(path, in_ext), key=lambda p: int(p.stem))
    try:
        # XXX images MUST be converted to adaptive color palette otherwise resulting gif has very bad quality
        imgs = [Image.open(f).convert('P', palette=Image.ADAPTIVE) for f in files]
    except OSError as e:
        print_warning(e)
        return

    if len(imgs) > 0:
        if size is not None and size != imgs[0].size:
            imgs = list(map(lambda i: resize(i, size=size), imgs))
        tpf = int(total_sec * 1000 / len(files))
        imgs[0].save(path.parent / name, optimize=False, save_all=True, append_images=imgs[1:], duration=tpf, loop=0)


def draw_border(img, color, width):
    a = np.asarray(img)
    for k in range(width):
        a[k, :] = color
        a[-k-1, :] = color
        a[:, k] = color
        a[:, -k-1] = color
    return Image.fromarray(a)


def pad_center(img_tensor, out_h, out_w):
    in_h, in_w = img_tensor.shape[2:]
    pad_l, pad_t = (out_w - in_w) // 2, (out_h - in_h) // 2
    return F.pad(img_tensor, (pad_l, out_w - in_w - pad_l, pad_t, out_h - in_h - pad_t), value=0)


def align_img(
    original_img,
    affine_tsf,
    bkg_color=(255, 60, 0),
    only_translate=True,
    out_size=None
):
    """
    Align image by undoing learned affine transformation at original scale.

    Args:
        original_img: PIL Image at original size
        affine_tsf: Learned affine transformation matrix [2, 3]
        bkg_color: Background color tuple (R, G, B)
        only_translate: If True, only undo translation; if False, undo full affine
        out_size: Output image size (w, h). If None, uses original image size

    Returns:
        PIL Image: Aligned image with alpha channel, size = out_size
    """
    img_tensor = ToTensor()(original_img).unsqueeze(0)

    if out_size is None:
        out_w, out_h = original_img.size
    else:
        out_w, out_h = out_size

    padded_img = pad_center(img_tensor, out_h, out_w)
    if only_translate:
        inverse_affine = torch.tensor([
            [1., 0., -affine_tsf[0, 2]],
            [0., 1., -affine_tsf[1, 2]]
        ], dtype=torch.float32)
    else:
        inverse_affine = invert_tsf(affine_tsf)

    grid = F.affine_grid(inverse_affine.unsqueeze(0), padded_img.shape, align_corners=False)
    aligned_tensor = F.grid_sample(padded_img, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    mask = (aligned_tensor.sum(dim=1, keepdim=True) > 1e-6).float()
    bkg_tensor = torch.tensor(bkg_color, dtype=torch.float32).view(1, 3, 1, 1) / 255.0
    result_tensor = aligned_tensor * mask + bkg_tensor * (1 - mask)

    result_with_alpha = torch.cat([result_tensor, mask], dim=1).squeeze(0).clamp(0, 1)
    return Image.fromarray((result_with_alpha.permute(1, 2, 0).numpy() * 255).astype(np.uint8), 'RGBA')


def combine_tsf(tsf1, tsf2):
    """Combine two affine transformations: result = tsf2 @ tsf1"""
    t1_homo = torch.cat([tsf1, torch.tensor([[0., 0., 1.]])], dim=0)
    t2_homo = torch.cat([tsf2, torch.tensor([[0., 0., 1.]])], dim=0)

    combined_homo = torch.matmul(t2_homo, t1_homo)
    return combined_homo[:2, :]


def invert_tsf(transform):
    """Invert an affine transformation matrix."""
    l = transform[:, :2]  # Linear part [2, 2]
    t = transform[:, 2:]  # Translation part [2, 1]

    l_inv = torch.inverse(l)
    return torch.cat([l_inv, -torch.matmul(l_inv, t)], dim=1)


class ImageResizer:
    """Resize images from a given input directory, keeping aspect ratio or not."""
    def __init__(self, input_dir, output_dir, size, in_ext=IMG_EXTENSIONS, out_ext='jpg', keep_aspect_ratio=True,
                 resample=Image.LANCZOS, fit_inside=True, rename=False, verbose=True):
        self.input_dir = coerce_to_path_and_check_exist(input_dir)
        self.files = get_files_from_dir(input_dir, valid_extensions=in_ext, recursive=True, sort=True)
        self.output_dir = coerce_to_path_and_create_dir(output_dir)
        self.out_extension = out_ext
        self.resize_func = partial(resize, size=size, keep_aspect_ratio=keep_aspect_ratio, resample=resample,
                                   fit_inside=fit_inside)
        self.rename = rename
        self.name_size = int(np.ceil(np.log10(len(self.files))))
        self.verbose = verbose

    def run(self):
        for k, filename in enumerate(self.files):
            if self.verbose:
                print_info('Resizing and saving {}'.format(filename))
            img = Image.open(filename).convert('RGB')
            img = self.resize_func(img)
            name = str(k).zfill(self.name_size) if self.rename else filename.stem
            img.save(self.output_dir / '{}.{}'.format(name, self.out_extension))
