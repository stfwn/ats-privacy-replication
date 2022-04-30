import random

from typing import List
from PIL.Image import Image

from thirdparty.autoaugment import SubPolicy as Transformation


EMPTY_TRANSFORMATION = -1


class Policy:
    def __init__(self, transformations: List[Transformation]):
        self.transformations = transformations

    def __call__(self, img: Image) -> Image:
        for transform in self.transformations:
            img = transform(img)
        return img

    @classmethod
    def from_indices(cls, transformations: List[int]):
        transformations = [TRANSFORMATIONS[t] for t in transformations]
        return cls(transformations)


class MetaPolicy:
    def __init__(self, policies: List[Policy]):
        self.policies = policies

    def __call__(self, img: Image) -> Image:
        policy = random.choice(self.policies)
        img = policy(img)
        return img

    @classmethod
    def from_indices(cls, policies: List[List[int]]):
        policies = [Policy.from_indices(p) for p in policies]
        return cls(policies)


TRANSFORMATIONS = [
    Transformation(0.1, "invert", 7),
    Transformation(0.2, "contrast", 6),
    Transformation(0.7, "rotate", 2),
    Transformation(0.3, "translateX", 9),
    Transformation(0.8, "sharpness", 1),
    Transformation(0.9, "sharpness", 3),
    Transformation(0.5, "shearY", 2),
    Transformation(0.7, "translateY", 2),
    Transformation(0.5, "autocontrast", 5),
    Transformation(0.9, "equalize", 2),
    Transformation(0.2, "shearY", 5),
    Transformation(0.3, "posterize", 5),
    Transformation(0.4, "color", 3),
    Transformation(0.6, "brightness", 5),
    Transformation(0.3, "sharpness", 9),
    Transformation(0.7, "brightness", 9),
    Transformation(0.6, "equalize", 5),
    Transformation(0.5, "equalize", 1),
    Transformation(0.6, "contrast", 7),
    Transformation(0.6, "sharpness", 5),
    Transformation(0.7, "color", 5),
    Transformation(0.5, "translateX", 5),
    Transformation(0.3, "equalize", 7),
    Transformation(0.4, "autocontrast", 8),
    Transformation(0.4, "translateY", 3),
    Transformation(0.2, "sharpness", 6),
    Transformation(0.9, "brightness", 6),
    Transformation(0.2, "color", 8),
    Transformation(0.5, "solarize", 0),
    Transformation(0.0, "invert", 0),
    Transformation(0.2, "equalize", 0),
    Transformation(0.6, "autocontrast", 0),
    Transformation(0.2, "equalize", 8),
    Transformation(0.6, "equalize", 4),
    Transformation(0.9, "color", 5),
    Transformation(0.6, "equalize", 5),
    Transformation(0.8, "autocontrast", 4),
    Transformation(0.2, "solarize", 4),
    Transformation(0.1, "brightness", 3),
    Transformation(0.7, "color", 0),
    Transformation(0.4, "solarize", 1),
    Transformation(0.9, "autocontrast", 0),
    Transformation(0.9, "translateY", 3),
    Transformation(0.7, "translateY", 3),
    Transformation(0.9, "autocontrast", 1),
    Transformation(0.8, "solarize", 1),
    Transformation(0.8, "equalize", 5),
    Transformation(0.1, "invert", 0),
    Transformation(0.7, "translateY", 3),
    Transformation(0.9, "autocontrast", 1),
]

NUM_TRANSFORMATIONS = len(TRANSFORMATIONS)
