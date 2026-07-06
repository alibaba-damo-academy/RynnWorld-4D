"""Smoke-test the RynnWorld4D deployment server over websocket.

Sends a raw 720x1280 head image + random 54-d state and checks the returned
action chunk shape/dtype. Run AFTER serve_rynnworld4d_policy.py is up.

    python smoke_test_serve.py --host 127.0.0.1 --port 8099
"""
import argparse
import sys
from pathlib import Path

import numpy as np

OPENPI_ROOT = Path(__file__).resolve().parent / "third_party" / "Openpi_damo"
sys.path.insert(0, str(OPENPI_ROOT / "packages" / "openpi-client" / "src"))

from openpi_client import websocket_client_policy  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8099)
    args = p.parse_args()

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print("server metadata:", client.get_server_metadata())

    obs = {
        "observation/state": np.random.randn(54).astype(np.float32),
        "observation/image": np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "prompt": "Pick-Place",
    }

    import time
    t0 = time.time()
    result = client.infer(obs)
    dt = (time.time() - t0) * 1000
    actions = result["actions"]
    print(f"infer OK in {dt:.1f} ms")
    print("actions shape:", np.asarray(actions).shape, "dtype:", np.asarray(actions).dtype)
    print("action[0][:5]:", np.asarray(actions)[0][:5])
    if "server_timing" in result:
        print("server_timing:", result["server_timing"])


if __name__ == "__main__":
    main()
