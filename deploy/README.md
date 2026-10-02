# Deploy PetSnap on Render with private model weights

For Railway, follow [the Railway deployment guide](RAILWAY.md) instead.

Docker installation on your computer is optional. Render builds the repository's
Dockerfile. This setup uses **AWS S3 private storage** and downloads weights at
**container startup**, not during the image build. No weights or credentials
are copied into the image. Nothing has been uploaded or deployed yet.

## 1. Put the checkpoint in private storage

In AWS, create an S3 bucket in a region you choose. Keep **Block all public
access** enabled and retain default encryption. Upload your existing
`models/best_model.pth` as `petsnap/v1/best_model.pth`. Do not upload the dataset.
The labels remain in Git; `deploy/model-manifest.json` binds their order to
this specific checkpoint.

The manifest records the current checkpoint's size and SHA-256. For this release:

```text
171613443 bytes
db8f1bde6f301f561b82125fb9cb8f6bbe4bd3021880f1ee8b1b1d7aa1924025
```

Create a dedicated deployment identity with permission to read only that object.
Use `deploy/s3-read-policy.example.json`, substituting your bucket name and key.
The example grants no listing, uploading, or deletion rights. Use this identity's
access key in Render; do not use your AWS root credentials. AWS credentials must
never be committed, embedded in the frontend, or passed as Docker build arguments.
S3 storage and transfer may incur separate charges from Render.

Optional: enable S3 versioning and set `PETSNAP_MODEL_VERSION_ID` in Render to
pin an exact object version. In that case grant `s3:GetObjectVersion` for the
same object instead of `s3:GetObject`. The checksum is required either way.
For customer-managed KMS encryption, the identity also needs the appropriate
`kms:Decrypt` permission; default S3-managed encryption avoids that extra setup.

## 2. Commit and push the deployment files

Commit the Dockerfile, `.dockerignore`, `render.yaml`, and `deploy/` files along
with the current application and labels. Keep weights ignored by Git.
The Docker context uses an allowlist to exclude weights, `.env` files, the
training dataset, `.runtime`, notebooks, and unrelated project files.

## 3. Create a Render Blueprint

In Render, choose **New → Blueprint**, connect your GitHub account, select the
PetSnap repository and your deployment branch, and use the root `render.yaml`.
Review the proposed service and cost before creating it. The template selects
one `1c-2g` instance (1 CPU / 2 GB RAM), Docker, and `/health` for readiness.
This is an initial sizing recommendation, not a measured memory guarantee.

Render prompts for these values because the Blueprint uses `sync: false`:

- `PETSNAP_MODEL_BUCKET`: your private bucket name, without `s3://`.
- `PETSNAP_MODEL_KEY`: `petsnap/v1/best_model.pth`, or the actual object key.
- `AWS_DEFAULT_REGION`: the bucket's region, for example `us-east-1`.
- `AWS_ACCESS_KEY_ID`: the dedicated read-only identity's access key ID.
- `AWS_SECRET_ACCESS_KEY`: its secret access key.

Temporary AWS credentials also require `AWS_SESSION_TOKEN` and must be refreshed
before they expire. Enter secrets directly in Render's Environment settings.
Do not put the 172 MB checkpoint in a Render secret file: secret files are for
small configuration values, not model storage.

Automatic deploys are disabled initially. Blueprint creation still triggers
the first deployment. For later changes, manually deploy the chosen commit
until the initial checks are complete. Environment changes in the dashboard
should use **Save and deploy** when only runtime settings have changed.

The Dockerfile pins direct Python dependencies and a compatible CPU-only
PyTorch/torchvision pair. Transitive dependencies and the base image tag are
not fully locked. Model download credentials are only consumed at runtime.

## 4. Configure proxy trust before sharing

`PETSNAP_TRUSTED_PROXY_IPS` starts empty: forwarding headers are ignored. This
prevents visitors from choosing a fake rate-limit identity, but requests through
a shared proxy will share that proxy's 10-attempt-per-minute quota.

Obtain the actual incoming proxy addresses/CIDRs for your Render service from
the provider; **outbound service IP ranges are not the incoming proxy ranges**.
Set `PETSNAP_TRUSTED_PROXY_IPS` to the verified comma-separated addresses/CIDRs
and redeploy. Wildcards are rejected by the entrypoint. Verify rate-limit
behavior from two separate networks before sharing. If the provider cannot
supply an appropriate trust configuration, keep the conservative shared quota
or implement provider-supported edge limiting before advertising per-IP limits.

Run one worker and one replica: the current rate limits are in memory, per
process. Multiple replicas need shared limiting. Keep any unused legacy API
routes inaccessible through an unset `DOG_CLASSIFIER_API_KEY` (the default).
No API key is needed for the browser demo. No CORS variable is needed when the
page and API share the Render service.

## 5. Check the first deployment

The container starts by downloading the private object over the AWS SDK's TLS
connection. It checks the expected byte count and SHA-256, then atomically
publishes it to `/app/model-cache/best_model.pth`. Incomplete or incorrect
downloads never reach PyTorch. Errors are redacted in startup logs.

Startup then replaces the launcher with one Uvicorn worker on `0.0.0.0:$PORT`.
The server becomes ready only after the model has loaded. Render should show
the health check as passing. Visit the assigned `onrender.com` URL and check:

- `/health` reports `ready` and 120 classes.
- All three sample photos produce predictions, and phone uploads work.
- Invalid images, large uploads, busy responses, and rate limits behave correctly.
- `/models/best_model.pth` and `/model-cache/best_model.pth` return 404.
- CPU, memory, startup time, and prediction latency are acceptable under light load.
- The service still starts after a restart using only its deployed settings.

No persistent disk is required. The cache may be discarded on restart or deploy,
so storage credentials must remain valid for future downloads. Use a new object
key for a new model, update the manifest and matching labels together, and test
before deploying. Keep old private objects so rollback remains possible.

## Optional local container test

With Docker Desktop running Linux containers, build without any secrets:

```sh
docker build -t petsnap-demo .
```

To test your existing local weights without any cloud access, run from the
repository root (PowerShell):

```powershell
docker run --rm -p 8018:10000 --mount "type=bind,source=$((Get-Location).Path)\models,target=/private-models,readonly" -e PETSNAP_MODEL_PATH=/private-models/best_model.pth petsnap-demo
```

Open http://127.0.0.1:8018. The startup script verifies the mounted checkpoint
and packaged labels and skips downloading. This does not test AWS permissions.
For a real storage test, supply runtime values through an ignored `.env.local`
file with `docker run --rm --env-file .env.local -p 8018:10000 petsnap-demo`.

Local unit tests for the download guard need no AWS account, Docker, or network:

```sh
python -m unittest discover -s tests -p test_deployment.py -v
```

## References

- https://render.com/docs/docker
- https://render.com/docs/blueprint-spec
- https://render.com/docs/configure-environment-variables
- https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-control-block-public-access.html
- https://pytorch.org/get-started/previous-versions/
