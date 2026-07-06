import dataclasses
import numpy as np
import einops

from openpi import transforms
from openpi.models import model as _model

def make_umi_example() -> dict:
    """
    Creates a random input example for the UMI policy.
    This example is aligned with UMIInputs / UMIOutputs.
    """
    T = 4  # action chunk length (example)

    return {
        # ---------- state ----------
        # Example: dual-arm state
        # [x_l, y_l, z_l, qx_l, qy_l, qz_l, qw_l, gripper_l,
        #  x_r, y_r, z_r, qx_r, qy_r, qz_r, qw_r, gripper_r]
        "observation/state": np.random.rand(16).astype(np.float32),

        # ---------- images ----------
        "observation/image": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),

        # Optional wrist images (can be removed to test padding logic)
        "observation/left_wrist_image": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/right_wrist_image": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),

        # ---------- actions (DELTA) ----------
        # shape: (T, action_dim)
        # Example: dual-arm delta actions
        "actions": np.random.randn(T, 16).astype(np.float32),

        # ---------- language ----------
        "prompt": "pick up the object and place it on the table",
    }

def _parse_image(image) -> np.ndarray:
    """
    Ensure image is uint8 (H, W, C)
    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UMIInputs(transforms.DataTransformFn):
    """
    UMI policy input transform.

    Assumptions:
    - dataset action is DELTA (no extra delta transform)
    - dual-arm: left + right + gripper
    - images may be missing -> zero padding
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # ---------- state ----------
        # example:
        # [
        #   x_l, y_l, z_l, qx_l, qy_l, qz_l, qw_l, gripper_l,
        #   x_r, y_r, z_r, qx_r, qy_r, qz_r, qw_r, gripper_r,
        # ]

        state = np.asarray(data["observation/state"])

        # ---------- images ----------
        #以left wrist为例
        left_wrist = _parse_image(data["observation/images/left_wrist"])

        # Optional front images (UMI often missing)！！
        base_image = (
            _parse_image(data["observation/images/front"])
            if "observation/images/front" in data
            else np.zeros_like(left_wrist)
        )
        right_wrist = (
            _parse_image(data["observation/images/right_wrist"])
            if "observation/images/right_wrist" in data
            else np.zeros_like(left_wrist)
        )

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # ---------- actions (DELTA, passed through) ----------
        if "actions" in data:
            # shape: (T, action_dim)
            inputs["actions"] = np.asarray(data["actions"])

        # ---------- language prompt ----------
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs
@dataclasses.dataclass(frozen=True)
class UMIOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :16])}
