import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_tianji_wuji_example() -> dict:
    """Creates a random input example for the Tianji Wuji policy.

    Dual arm + dexterous hand: left_arm 7DOF + left_hand 5x4=20 + right_arm 7DOF + right_hand 5x4=20 = 54 dims.
    """
    return {
        "observation/state": np.random.rand(54),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "flip the steak",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class TianjiWujiInputs(transforms.DataTransformFn):
    """Tianji Wuji dexterous hand policy input transform.

    State layout (54 dims):
        Left arm joints:    [left_arm_joint_1..7]          idx 0-6
        Left finger joints: [left_finger1..5_joint1..4]    idx 7-26
        Right arm joints:   [right_arm_joint_1..7]         idx 27-33
        Right finger joints: [right_finger1..5_joint1..4]  idx 34-53
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": data["observation/state"],
        }

        if "observation/image" in data:
            base_image = _parse_image(data["observation/image"])
            left_wrist = _parse_image(data["observation/left_wrist_image"])
            right_wrist = _parse_image(data["observation/right_wrist_image"])
            inputs["image"] = {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            }
            inputs["image_mask"] = {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


def _center_crop_resize(image: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Center crop to square, then resize to target_size x target_size.

    Matches the resize method used in Improved-3D-Diffusion-Policy.
    """
    h, w = image.shape[:2]
    crop_size = min(h, w)
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    image = image[top:top + crop_size, left:left + crop_size]
    # Use PIL for resize (bilinear interpolation)
    from PIL import Image
    image_pil = Image.fromarray(image)
    image_pil = image_pil.resize((target_size, target_size), Image.BILINEAR)
    return np.asarray(image_pil)


@dataclasses.dataclass(frozen=True)
class TianjiWujiHeadOnlyInputs(transforms.DataTransformFn):
    """Tianji Wuji policy input transform using only the head camera.

    Uses center crop + resize to 224x224 for the head camera.
    Wrist cameras are masked out (not provided).

    State layout (54 dims):
        Left arm joints:    [left_arm_joint_1..7]          idx 0-6
        Left finger joints: [left_finger1..5_joint1..4]    idx 7-26
        Right arm joints:   [right_arm_joint_1..7]         idx 27-33
        Right finger joints: [right_finger1..5_joint1..4]  idx 34-53
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": data["observation/state"],
        }

        # Only head camera - center crop + resize to 224x224
        if "observation/image" in data:
            base_image = _parse_image(data["observation/image"])
            base_image = _center_crop_resize(base_image)

            # Create blank black images for wrist cameras (masked out)
            dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)

            inputs["image"] = {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": dummy_image,
                "right_wrist_0_rgb": dummy_image,
            }
            inputs["image_mask"] = {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class TianjiWujiOutputs(transforms.DataTransformFn):
    """Tianji Wuji policy output transform (inference only).

    Extracts the first 54 dims = left_arm 7 + left_hand 20 + right_arm 7 + right_hand 20.
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :54])}
