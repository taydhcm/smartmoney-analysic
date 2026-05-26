"""
scripts/set_ssi_token.py
Helper script de cap nhat SSI Bearer token tu browser F12.

Huong dan:
  1. Mo iboard.ssi.com.vn, dang nhap
  2. F12 -> Network -> bat ky request den iboard-tapi.ssi.com.vn
  3. Copy gia tri "Authorization: Bearer <token>" (chi phan token, khong co chu "Bearer")
  4. Chay: py -3.12 scripts/set_ssi_token.py <token>

Token co TTL 8 gio (SSI JWT HS256, channel=web, systemType=iboard).
"""
import sys
import os
import base64
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"

sys.path.insert(0, str(ROOT))


def decode_jwt_exp(token: str) -> int:
    """Decode JWT payload va tra ve exp timestamp (0 neu khong co)."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return 0
        padding = "==" * (4 - len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + padding))
        return int(payload.get("exp", 0))
    except Exception:
        return 0


def update_env_token(token: str) -> None:
    """Cap nhat hoac them SSI_BEARER_TOKEN vao .env."""
    lines = []
    found = False

    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("SSI_BEARER_TOKEN="):
                lines.append(f"SSI_BEARER_TOKEN={token}")
                found = True
            else:
                lines.append(line)

    if not found:
        lines.append(f"SSI_BEARER_TOKEN={token}")

    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if len(sys.argv) < 2:
        # Interactive mode: nhap token tu stdin
        print("SSI Bearer Token Setup")
        print("=" * 50)
        print("Huong dan:")
        print("  1. Mo iboard.ssi.com.vn va dang nhap")
        print("  2. F12 -> Network -> mo request bat ky den iboard-tapi.ssi.com.vn")
        print("  3. Copy gia tri 'authorization' header (phan sau chu 'Bearer ')")
        print()
        print("Dan token vao day (Ctrl+V roi Enter):")
        token = input("> ").strip()
    else:
        token = sys.argv[1].strip()
        # Loai bo prefix "Bearer " neu nguoi dung copy ca prefix
        if token.lower().startswith("bearer "):
            token = token[7:].strip()

    if not token:
        print("ERROR: Token trong. Thoat.")
        sys.exit(1)

    # Validate JWT format
    parts = token.split(".")
    if len(parts) != 3:
        print("WARNING: Day khong phai JWT hop le (phai co 3 phan ngan cach boi dau .)")
        print(f"  Nhan duoc: {token[:50]}...")

    # Kiem tra expiry
    exp = decode_jwt_exp(token)
    if exp:
        from datetime import datetime
        exp_dt = datetime.fromtimestamp(exp)
        now = time.time()
        if exp < now:
            print(f"WARNING: Token nay da het han luc {exp_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            print("         Vui long lay token moi tu browser.")
        else:
            remaining_h = (exp - now) / 3600
            print(f"Token con hieu luc den: {exp_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Thoi gian con lai: {remaining_h:.1f} gio")

    # Luu vao .env
    update_env_token(token)
    print(f"\nDa luu SSI_BEARER_TOKEN vao {ENV_PATH}")
    print("Token se duoc su dung tu dong khi fetch investor flow data.")
    print()
    print("Khi token het han (sau 8 gio), chay lai lenh nay de cap nhat.")


if __name__ == "__main__":
    main()
