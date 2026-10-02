"""Interactive, local-only upload to private Railway storage. Never saves credentials."""
import argparse
import getpass
import json
from pathlib import Path
import tempfile
import warnings
from urllib.parse import urlsplit

if __package__:
    from .download_model import ROOT, fetch_model, sha256_file, verify_labels
else:
    from download_model import ROOT, fetch_model, sha256_file, verify_labels


def validate_endpoint(value):
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/")):
        raise ValueError("Use the base HTTPS endpoint from Railway's Credentials tab.")
    return value.rstrip("/")


def verify_local(path, manifest):
    if not path.is_file() or path.stat().st_size != manifest["size_bytes"]:
        raise ValueError("Local model is missing or does not match the expected size.")
    if sha256_file(path) != manifest["sha256"]:
        raise ValueError("Local model does not match the deployment checksum. Nothing was uploaded.")
    verify_labels(ROOT / "models/classes.json", manifest["labels_sha256"])


def upload_and_verify(client, path, bucket, key, manifest):
    # Existing objects are never replaced by this helper.
    from botocore.exceptions import ClientError
    exists = False
    try:
        client.head_object(Bucket=bucket, Key=key)
        exists = True
    except ClientError as exc:
        if str(exc.response.get("Error", {}).get("Code")) not in ("404", "NoSuchKey", "NotFound"):
            raise
    if not exists:
        print("Uploading the checkpoint privately. This may take a few minutes...", flush=True)
        client.upload_file(str(path), bucket, key)
    else:
        print("An object already exists at this key. Verifying it without replacing it...", flush=True)
    print("Reading the stored object back to verify its full checksum...", flush=True)
    with tempfile.TemporaryDirectory(prefix="petsnap-verify-") as directory:
        fetch_model(client, bucket, key, Path(directory) / "model.pth", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True, help="Actual bucket name from Railway Credentials")
    parser.add_argument("--key", default="petsnap/v1/best_model.pth")
    args = parser.parse_args()
    client = None
    try:
        import boto3
        from botocore.config import Config

        path = ROOT / "models/best_model.pth"
        manifest = json.loads((ROOT / "deploy/model-manifest.json").read_text(encoding="utf-8"))
        verify_local(path, manifest)
        endpoint = validate_endpoint(input("Railway ENDPOINT (base HTTPS URL): ").strip())
        region = input("Railway REGION [auto]: ").strip() or "auto"
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            access_key = getpass.getpass("Railway ACCESS_KEY_ID (hidden): ")
            secret_key = getpass.getpass("Railway SECRET_ACCESS_KEY (hidden): ")
        if not access_key or not secret_key:
            raise ValueError("Both credentials are required.")
        # An explicit session avoids picking up unrelated local AWS profiles/tokens.
        session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key,
                                region_name=region)
        client = session.client("s3", endpoint_url=endpoint,
                                config=Config(connect_timeout=10, read_timeout=60,
                                              retries={"mode": "standard", "total_max_attempts": 3},
                                              s3={"addressing_style": "virtual"},
                                              request_checksum_calculation="when_required"))
        upload_and_verify(client, path, args.bucket, args.key, manifest)
        print(f"Verified. Set PETSNAP_MODEL_BUCKET={args.bucket}")
        print(f"Set PETSNAP_MODEL_KEY={args.key}. The checkpoint is ready for deployment.")
    except ImportError:
        raise SystemExit("Install boto3 in your local upload environment first; see deploy/RAILWAY.md.") from None
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    except (KeyboardInterrupt, EOFError):
        raise SystemExit("Upload cancelled.") from None
    except getpass.GetPassWarning:
        raise SystemExit("Run this helper in an interactive terminal that supports hidden password input.") from None
    except Exception:
        raise SystemExit("Upload or verification failed. Check Railway credentials, endpoint, region, and object key. No existing object was replaced; any new upload may need verification. Private error details were omitted.") from None
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
