"""Fetch a private S3 object at runtime and verify it before any torch.load call."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent


class ModelDownloadError(RuntimeError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_labels(path, expected):
    labels = json.loads(Path(path).read_text(encoding="utf-8"))
    # Canonical JSON keeps the fingerprint stable across Windows/Linux line endings.
    canonical = json.dumps(labels, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != expected:
        raise ModelDownloadError("Breed labels do not match the deployment manifest.")


def fetch_model(client, bucket, key, target, manifest, version_id=None):
    """Bound streaming downloads; only atomically publish a fully verified file."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    args = {"Bucket": bucket, "Key": key}
    if version_id:
        args["VersionId"] = version_id
    temporary = None
    response = client.get_object(**args)
    body = response["Body"]
    try:
        if response.get("ContentLength") != manifest["size_bytes"]:
            raise ModelDownloadError("Private model has an unexpected size.")
        digest = hashlib.sha256()
        total = 0
        deadline = time.monotonic() + 300
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".model-", suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            for chunk in body.iter_chunks(chunk_size=1024 * 1024):
                if time.monotonic() > deadline:
                    raise ModelDownloadError("Private model download timed out.")
                total += len(chunk)
                if total > manifest["size_bytes"]:
                    raise ModelDownloadError("Private model exceeds the expected size.")
                output.write(chunk)
                digest.update(chunk)
        if total != manifest["size_bytes"] or digest.hexdigest() != manifest["sha256"]:
            raise ModelDownloadError("Private model checksum verification failed.")
        os.replace(temporary, target)
        temporary = None
    finally:
        body.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ensure_model():
    manifest = json.loads((ROOT / "deploy/model-manifest.json").read_text(encoding="utf-8"))
    target = Path(os.environ.get("PETSNAP_MODEL_PATH", ROOT / "model-cache/best_model.pth"))
    labels = Path(os.environ.get("PETSNAP_CLASSES_PATH", ROOT / "models/classes.json"))
    verify_labels(labels, manifest["labels_sha256"])
    if target.is_file() and target.stat().st_size == manifest["size_bytes"]:
        if sha256_file(target) == manifest["sha256"]:
            print("Verified cached PetSnap checkpoint.", flush=True)
            return target
    bucket = os.environ.get("PETSNAP_MODEL_BUCKET")
    key = os.environ.get("PETSNAP_MODEL_KEY")
    if not bucket or not key:
        raise ModelDownloadError("Set PETSNAP_MODEL_BUCKET and PETSNAP_MODEL_KEY in the hosting environment.")
    # Import only when a download is needed, allowing offline cache verification.
    import boto3
    from botocore.config import Config

    client = boto3.client("s3", config=Config(connect_timeout=10, read_timeout=30,
                                             retries={"mode": "standard", "total_max_attempts": 3}))
    try:
        fetch_model(client, bucket, key, target, manifest, os.environ.get("PETSNAP_MODEL_VERSION_ID"))
    finally:
        client.close()
    print("Private PetSnap checkpoint downloaded and verified.", flush=True)
    return target


def main():
    try:
        return ensure_model()
    except ModelDownloadError as exc:
        raise SystemExit(str(exc)) from None
    except Exception:
        # Provider exceptions may include private bucket names, URLs, or credentials.
        raise SystemExit("Model setup failed. Check private storage access, region, and model configuration.") from None


if __name__ == "__main__":
    main()
