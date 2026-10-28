"""Two-step GitHub device-flow login for the Caltech SoCal 28-bus dataset API.
EXPLORATORY tooling, no tests.

The dataset client's own login blocks while it polls, which does not fit a
non-interactive shell. This splits it: `start` prints a user code for the user to enter at
https://github.com/login/device (valid ~15 min); `finish` makes ONE token request.
Nothing here prints the device code or any token. Credentials live outside the repo.

Setup (once):
  git clone --depth 1 https://github.com/caltech-netlab/digital-twin-dataset ~/.cache/caltech_dt
  pip install requests stream-unzip tqdm pyperclip
  mkdir -m 700 -p ~/.config/caltech_creds
Usage:  python -m island_detectability_map.caltech_login start | finish | whoami
Env:    CALTECH_CLIENT_DIR (default ~/.cache/caltech_dt), CALTECH_CREDS_DIR (default ~/.config/caltech_creds)

Origin: verification@67698f42
"""
from __future__ import annotations

import json, os, sys, time
from pathlib import Path

CLIENT_DIR = os.environ.get("CALTECH_CLIENT_DIR", os.path.expanduser("~/.cache/caltech_dt"))
CREDS = Path(os.environ.get("CALTECH_CREDS_DIR", os.path.expanduser("~/.config/caltech_creds")))
PENDING, CREDFILE = CREDS / "pending_device.json", CREDS / "dataset_api_credentials.json"


def client():
    sys.path.insert(0, CLIENT_DIR)
    from dataset_api_client import DatasetApiClient
    return DatasetApiClient(credentials_file=str(CREDFILE))


def main(cmd: str):
    c = client()
    from dataset_api_client import GitHubTokens, GitHubVerification
    if cmd == "start":
        v = GitHubVerification.request(c.github_client_id)
        PENDING.write_text(json.dumps(v.__dict__)); os.chmod(PENDING, 0o600)
        exp = time.strftime("%H:%M UTC", time.gmtime(v.issued_at + v.expires_in))
        print(f"USER CODE: {v.user_code}\nENTER AT:  {v.verification_uri}\nEXPIRES:   {exp}")
    elif cmd == "finish":
        v = GitHubVerification(**json.loads(PENDING.read_text()))
        if v.expired:
            print("EXPIRED: run start again"); return 1
        try:
            t = GitHubTokens.request(c.github_client_id, v.device_code, str(CREDFILE))
        except RuntimeError as e:
            print("ERROR:", str(e)[:200]); return 2
        if t is None:
            print("PENDING: code not yet entered"); return 0
        os.chmod(CREDFILE, 0o600); PENDING.unlink()
        print(f"OK: tokens saved (access {t.expires_in // 3600} h, refresh {t.refresh_token_expires_in // 86400} d)")
    elif cmd == "whoami":
        import requests
        r = requests.get(f"{c.base_url}/user", headers=c.auth_header, timeout=60)
        print(r.status_code, r.json().get("github_username") if r.ok else r.text[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
