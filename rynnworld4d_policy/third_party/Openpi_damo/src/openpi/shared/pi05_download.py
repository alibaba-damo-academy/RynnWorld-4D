import os
from pathlib import Path

# Cache root for OpenPI downloads. Override with the OPENPI_DATA_HOME env var
# before running this script; otherwise a repo-local ./cache/openpi dir is used.
os.environ.setdefault(
    "OPENPI_DATA_HOME",
    str(Path(__file__).resolve().parents[4] / "cache" / "openpi"),
)

# Disable TensorStore file locking (recommended when reading from GCS).
os.environ["TENSORSTORE_CONTEXT"] = '{"file_io_locking": false}'

from openpi.shared.download import maybe_download

def main():
    url = "gs://openpi-assets/checkpoints/pi0_base/params"

    local_path = maybe_download(
        url,
        # 如果是公开 bucket，一般不需要额外参数
        # anon=True  # 视环境而定
    )

    print("✅ Download finished")
    print("Local path:", local_path)

    # 简单 sanity check
    assert Path(local_path).exists()

if __name__ == "__main__":
    main()
