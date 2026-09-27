"""Runtime entrypoint: fetch private weights, then replace this process with Uvicorn."""
import os
from pathlib import Path
import sys

if __package__:
    from .download_model import main as prepare_model
else:
    from download_model import main as prepare_model


def main():
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    port = os.environ.get("PORT", "10000")
    if not port.isascii() or not port.isdigit() or not 1 <= int(port) <= 65535:
        raise SystemExit("PORT must be between 1 and 65535.")
    command = [sys.executable, "-m", "uvicorn", "api.api_server:app",
               "--host", "0.0.0.0", "--port", port, "--workers", "1"]
    # Set only verified proxy addresses; never infer them or trust arbitrary XFF.
    trusted = os.environ.get("PETSNAP_TRUSTED_PROXY_IPS", "").strip()
    if trusted:
        if "*" in trusted:
            raise SystemExit("Specify trusted proxy addresses explicitly, not a wildcard.")
        command += ["--proxy-headers", "--forwarded-allow-ips", trusted]
    else:
        command += ["--no-proxy-headers"]
    model = prepare_model()
    os.environ["PETSNAP_MODEL_PATH"] = str(model.resolve())
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
