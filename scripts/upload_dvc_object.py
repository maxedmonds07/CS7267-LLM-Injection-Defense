"""Upload one DVC-cached file to the R2 remote with small multipart parts.

DVC/s3fs always uses 50 MiB parts; on a slow uplink those requests stay open for minutes
and get cut off (IncompleteBody). This writes the same object to the same key DVC uses,
so `dvc status --cloud` / `dvc pull` see it as pushed.

Usage (repo root): make push-large DVC_FILE=path/to/file.dvc [PART_MIB=8]
"""

import sys
from pathlib import Path

import boto3
import yaml
from boto3.s3.transfer import TransferConfig
from dvc.repo import Repo

dvc_file = Path(sys.argv[1])
part_mib = int(sys.argv[2]) if len(sys.argv) > 2 else 8

(out,) = yaml.safe_load(dvc_file.read_text())["outs"]
md5, size = out["md5"], out["size"]
cache_path = Path(".dvc/cache/files/md5") / md5[:2] / md5[2:]
assert cache_path.stat().st_size == size, "cache file size doesn't match the .dvc file"

repo = Repo(".")
remote_name = repo.config["core"]["remote"]
remote = repo.config["remote"][remote_name]  # merged with .dvc/config.local (credentials)
bucket, _, prefix = remote["url"].removeprefix("s3://").partition("/")
key = f"{prefix.rstrip('/')}/files/md5/{md5[:2]}/{md5[2:]}"

s3 = boto3.client(
    "s3",
    endpoint_url=remote["endpointurl"],
    region_name=remote.get("region", "auto"),
    aws_access_key_id=remote["access_key_id"],
    aws_secret_access_key=remote["secret_access_key"],
)

sent = 0


def progress(n: int) -> None:
    global sent
    sent += n
    print(f"\r{sent / 2**20:7.1f} / {size / 2**20:.1f} MiB", end="", flush=True)


config = TransferConfig(
    multipart_threshold=part_mib * 2**20,
    multipart_chunksize=part_mib * 2**20,
    max_concurrency=1,
)
print(f"{cache_path} -> s3://{bucket}/{key} ({part_mib} MiB parts)")
s3.upload_file(str(cache_path), bucket, key, Config=config, Callback=progress)
print()

remote_size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
assert remote_size == size, f"remote size {remote_size} != {size}"
print(f"uploaded and verified: {remote_size} bytes")
