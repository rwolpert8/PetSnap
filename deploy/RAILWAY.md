# Railway deployment

Use the existing Dockerfile; `render.yaml` is not used by Railway.
The build also caches a public dog detector; no additional bucket upload or secrets
are needed for it. See [detector behavior and validation](../docs/dog-detection.md).
Create a private Railway bucket and configure these variables on the app service:

- `PETSNAP_MODEL_BUCKET`: actual bucket name from Credentials (includes suffix).
- `PETSNAP_MODEL_KEY`: `petsnap/v1/best_model.pth`.
- `AWS_ENDPOINT_URL`: base HTTPS endpoint from bucket Credentials, without the bucket name.
- `AWS_DEFAULT_REGION`: region from Credentials, often `auto`.
- `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`: Railway bucket credentials.
- `PORT`: `10000`.

Use Railway variable references for bucket credentials where possible. These
AWS-prefixed SDK variables connect to Railway; no AWS account is needed.
Leave `PETSNAP_MODEL_VERSION_ID` unset: Railway object versioning is unsupported.

## Upload from Windows PowerShell

Run in the repository root. The following creates an isolated upload environment:

```powershell
python -m venv .venv-upload
.\.venv-upload\Scripts\python.exe -m pip install boto3==1.43.93
.\.venv-upload\Scripts\python.exe deploy/upload_model.py --bucket petsnap-models-im9ciljwjj
```

Copy the endpoint, region, access key ID, and secret from the bucket's Credentials
tab into the interactive prompts. Both key prompts are hidden; paste and press
Enter even though no characters appear. Credentials are not saved to files or
included in command history by this script. Run locally in an interactive terminal,
not in chat, CI, a recorded terminal, or a deployment start command.

The helper verifies your local checkpoint against the committed manifest, uploads
to `petsnap/v1/best_model.pth`, then downloads it to a temporary file and checks the
full SHA-256. The temporary verification copy is removed. If an object already
exists, it is verified instead of uploaded again. A mismatched existing object
requires an intentional new key or a separate replacement; this helper stops.
Run only one uploader at a time for this object key.

If upload fails, confirm the bucket itself has been deployed/provisioned. An app
deployment is not needed to upload to an already provisioned bucket.

## Start the app

1. Commit and push the Railway compatibility changes, including virtual-hosted
   S3 addressing in `deploy/download_model.py`.
2. In the app service Settings, leave the custom start command empty (use Docker's
   CMD), set health-check path `/health`, and keep a single replica. Allow enough
   memory for the model; start with a 2 GB limit and measure usage.
3. Apply your variables and deploy. Look for `Private PetSnap checkpoint downloaded
   and verified.` followed by `Application startup complete.` in deployment logs.
4. Generate a domain under Settings → Networking; select target port `10000`.
5. Check `/health` and the three sample predictions.

The current proxy defaults intentionally ignore forwarded client addresses, so
visitors may share the proxy's rate quota. Verify provider-specific proxy trust
before claiming independent per-visitor limits; do not use a wildcard trust value.
The model is kept outside the static frontend directory and is not publicly served.

References: https://docs.railway.com/storage-buckets and
https://docs.railway.com/deployments/healthchecks
