# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Modified for sports-specific jersey augmentation.

from functools import partial

import imgaug.augmenters as iaa
import numpy as np

if not hasattr(np, "bool"):
    np.bool = bool

from PIL import ImageFilter, Image
from timm.data import auto_augment

from strhub.data import aa_overrides

aa_overrides.apply()

_OP_CACHE = {}


def _get_op(key, factory):
    try:
        op = _OP_CACHE[key]
    except KeyError:
        op = factory()
        _OP_CACHE[key] = op
    return op


def _get_param(level, img, max_dim_factor, min_level=1):
    max_level = max(min_level, max_dim_factor * max(img.size))
    return round(min(level, max_level))


def gaussian_blur(img, radius, **__):
    radius = _get_param(radius, img, 0.02)
    key = 'gaussian_blur_' + str(radius)
    op = _get_op(key, lambda: ImageFilter.GaussianBlur(radius))
    return img.filter(op)


def motion_blur(img, k, **__):
    k = _get_param(k, img, 0.08, 3) | 1
    key = 'motion_blur_' + str(k)
    op = _get_op(key, lambda: iaa.MotionBlur(k))
    return Image.fromarray(op(image=np.asarray(img)))


def gaussian_noise(img, scale, **_):
    scale = _get_param(scale, img, 0.25) | 1
    key = 'gaussian_noise_' + str(scale)
    op = _get_op(key, lambda: iaa.AdditiveGaussianNoise(scale=scale))
    return Image.fromarray(op(image=np.asarray(img)))


def poisson_noise(img, lam, **_):
    lam = _get_param(lam, img, 0.2) | 1
    key = 'poisson_noise_' + str(lam)
    op = _get_op(key, lambda: iaa.AdditivePoissonNoise(lam))
    return Image.fromarray(op(image=np.asarray(img)))


def _level_to_arg(level, _hparams, max):
    level = max * level / auto_augment._LEVEL_DENOM
    return level,


_RAND_TRANSFORMS = auto_augment._RAND_INCREASING_TRANSFORMS.copy()
_RAND_TRANSFORMS.remove('SharpnessIncreasing')
_RAND_TRANSFORMS.extend([
    'GaussianBlur',
    'PoissonNoise',
])
auto_augment.LEVEL_TO_ARG.update({
    'GaussianBlur': partial(_level_to_arg, max=4),
    'MotionBlur': partial(_level_to_arg, max=20),
    'GaussianNoise': partial(_level_to_arg, max=0.1 * 255),
    'PoissonNoise': partial(_level_to_arg, max=40)
})
auto_augment.NAME_TO_OP.update({
    'GaussianBlur': gaussian_blur,
    'MotionBlur': motion_blur,
    'GaussianNoise': gaussian_noise,
    'PoissonNoise': poisson_noise
})


def rand_augment_transform(magnitude=5, num_layers=3):
    hparams = {
        'rotate_deg': 30,
        'shear_x_pct': 0.9,
        'shear_y_pct': 0.2,
        'translate_x_pct': 0.10,
        'translate_y_pct': 0.30
    }
    ra_ops = auto_augment.rand_augment_ops(magnitude, hparams=hparams, transforms=_RAND_TRANSFORMS)
    choice_weights = [1. / len(ra_ops) for _ in range(len(ra_ops))]
    return auto_augment.RandAugment(ra_ops, num_layers, choice_weights)


class SportsJerseyAugment:
    """
    Conservative sports-specific training-time augmentation for jersey-number STR.

    Targets proposal failure modes:
    - motion blur
    - perspective distortion
    - partial occlusion / masking
    - brightness/contrast variation

    Designed to be milder than the proposal maxima on the first run so it is less likely
    to destroy already tiny crops.
    """
    def __init__(self):
        self.seq = iaa.Sequential([
            iaa.Sometimes(
                0.25,
                iaa.MotionBlur(k=(3, 11))
            ),
            iaa.Sometimes(
                0.20,
                iaa.PerspectiveTransform(scale=(0.03, 0.10), keep_size=True)
            ),
            iaa.Sometimes(
                0.30,
                iaa.OneOf([
                    iaa.Cutout(
                        nb_iterations=(1, 2),
                        size=(0.04, 0.12),
                        squared=False,
                        fill_mode="constant",
                        cval=(0, 255),
                    ),
                    iaa.CoarseDropout(
                        p=(0.01, 0.03),
                        size_percent=(0.04, 0.10),
                        per_channel=False
                    )
                ])
            ),
            iaa.Sometimes(
                0.30,
                iaa.Sequential([
                    iaa.Multiply((0.80, 1.20)),
                    iaa.LinearContrast((0.75, 1.30))
                ])
            ),
        ], random_order=True)

    def __call__(self, img):
        arr = np.asarray(img)
        aug = self.seq(image=arr)
        return Image.fromarray(aug)
