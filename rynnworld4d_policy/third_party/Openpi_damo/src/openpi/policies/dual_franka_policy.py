import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_franka_bimanual_example() -> dict:
    """Creates a random input example for the Franka Bimanual policy.
    双臂 Franka: 左臂7DOF  + 右臂7DOF + 1夹爪 + 1夹爪 = 16维
    """
    return {
        # [j0_l, j1_l, j2_l, j3_l, j4_l, j5_l, j6_l, 
        #  j0_r, j1_r, j2_r, j3_r, j4_r, j5_r, j6_r, gripper_l, gripper_r]
        "observation/state": np.random.rand(16),
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
class FrankaBimanualInputs(transforms.DataTransformFn):
    """
    双臂 Franka 策略输入变换。
    State 布局 (16维):
        左臂关节: [j0_l, j1_l, j2_l, j3_l, j4_l, j5_l, j6_l]  idx 0-6
        右臂关节: [j0_r, j1_r, j2_r, j3_r, j4_r, j5_r, j6_r]  idx 7-13
        夹爪:     [gripper_l, gripper_r]                         idx 14-15
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        left_wrist = _parse_image(data["observation/left_wrist_image"])
        right_wrist = _parse_image(data["observation/right_wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaBimanualOutputs(transforms.DataTransformFn):
    """
    双臂 Franka 策略输出变换 (仅推理时使用)。
    取前16维 = 左臂7 + 右臂7 + 左夹爪1 + 右夹爪1
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :16])}
