"""Check that an R2 API token can reach the DVC remote, without touching tracked data.

Tests the keys in .dvc/config.local, or R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY if both are
set, so a new token can be tried before it is stored or handed to a teammate.

--write also uploads, reads back and deletes a small object outside the DVC prefix.

Usage (repo root): make r2-check | make r2-check-write
"""

import os
import sys
import uuid

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from dvc.repo import Repo

PROBE_PREFIX = "_r2-check"

HINTS = {
    "AccessDenied": "the token is expired, revoked, or lacks this permission or bucket",
    "InvalidAccessKeyId": "the Access Key ID is wrong, or the token was deleted",
    "InvalidArgument": "the keys are malformed; check for a truncated or pasted-over value",
    "NoSuchBucket": "the bucket in .dvc/config doesn't exist in this account",
    "SignatureDoesNotMatch": "the Secret Access Key doesn't match the Access Key ID",
    "Unauthorized": "the token is expired, revoked, or invalid",
}


def describe(error: Exception) -> str:
    if isinstance(error, ClientError):
        code = error.response["Error"]["Code"]
        hint = HINTS.get(code)
        return f"{code}: {hint}" if hint else f"{code}: {error.response['Error'].get('Message', '')}"
    if isinstance(error, BotoCoreError):
        return f"{type(error).__name__}: couldn't reach R2 ({error})"
    return str(error)


def check_read(s3, bucket: str, prefix: str) -> None:
    """List the DVC prefix, then read one byte of its first object."""
    listing = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    print(f"  ok    list   s3://{bucket}/{prefix}")
    objects = listing.get("Contents", [])
    if not objects:
        print("  skip  read   the prefix is empty, so object reads are untested")
        return
    s3.get_object(Bucket=bucket, Key=objects[0]["Key"], Range="bytes=0-0")["Body"].read()
    print(f"  ok    read   {objects[0]['Key']}")


def check_write(s3, bucket: str) -> None:
    """Upload, read back and delete a probe object; always try to delete it."""
    key = f"{PROBE_PREFIX}/{uuid.uuid4().hex}"
    body = b"r2 token check"
    s3.put_object(Bucket=bucket, Key=key, Body=body)
    try:
        print(f"  ok    write  {key}")
        read_back = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        if read_back != body:
            raise ValueError(f"probe object {key} came back with different content")
        print("  ok    verify")
    finally:
        s3.delete_object(Bucket=bucket, Key=key)
        print("  ok    delete")


def main() -> int:
    repo = Repo(".")
    remote = repo.config["remote"][repo.config["core"]["remote"]]
    bucket, _, prefix = remote["url"].removeprefix("s3://").partition("/")

    env_id, env_secret = os.environ.get("R2_ACCESS_KEY_ID"), os.environ.get("R2_SECRET_ACCESS_KEY")
    if env_id and env_secret:
        key_id, secret, source = env_id, env_secret, "environment"
    else:
        key_id, secret = remote.get("access_key_id"), remote.get("secret_access_key")
        source = ".dvc/config.local"
    if not key_id or not secret:
        print("No R2 keys found. Run `make r2-credentials` or set both R2_* variables.")
        return 1

    print(f"Token ...{key_id[-4:]} from {source}")
    s3 = boto3.client(
        "s3",
        endpoint_url=remote["endpointurl"],
        region_name=remote.get("region", "auto"),
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        config=Config(connect_timeout=10, read_timeout=30, retries={"max_attempts": 1}),
    )

    try:
        check_read(s3, bucket, prefix)
        if "--write" in sys.argv[1:]:
            check_write(s3, bucket)
    except (ClientError, BotoCoreError, ValueError) as error:
        print(f"  FAIL  {describe(error)}")
        return 1

    print("Token works." if "--write" in sys.argv[1:] else "Read access works. Writes are untested; run `make r2-check-write`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
