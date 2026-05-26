"""
analytics/secret_manager.py
Machine-specific Fernet encryption cho credentials nhạy cảm.

Key được derive từ hostname + MAC address → chỉ giải mã được trên máy này.
Credentials lưu encrypted trong .env (đã gitignore) → an toàn khi push code.

Usage:
    from analytics.secret_manager import get_ssi_credentials

    account, password = get_ssi_credentials()
"""

from __future__ import annotations

import base64
import hashlib
import os
import socket
import uuid
from pathlib import Path

from utils.logger import get_logger

log = get_logger(__name__)

_SALT    = b"smartmoney-ssi-salt-v1"
_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


# ── Key derivation ────────────────────────────────────────────────────────────

def _derive_machine_key() -> bytes:
    """
    Derive Fernet-compatible key (32-byte URL-safe base64) từ machine identity.
    Deterministic trên cùng máy, không lưu key ra file.
    """
    machine_id = f"{socket.gethostname()}-{uuid.getnode()}"
    raw = hashlib.pbkdf2_hmac(
        "sha256",
        machine_id.encode("utf-8"),
        _SALT,
        iterations=200_000,
        dklen=32,
    )
    return base64.urlsafe_b64encode(raw)


def _fernet():
    from cryptography.fernet import Fernet
    return Fernet(_derive_machine_key())


# ── Public helpers ────────────────────────────────────────────────────────────

def encrypt_value(plaintext: str) -> str:
    """Encrypt plaintext string → URL-safe ciphertext string."""
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt_value(ciphertext: str) -> str:
    """Decrypt ciphertext string → plaintext. Raises on wrong machine/tampered."""
    return _fernet().decrypt(ciphertext.encode()).decode()


def get_ssi_credentials() -> tuple[str, str]:
    """
    Đọc và giải mã SSI FastConnect API credentials từ .env.

    Returns
    -------
    (consumerID, consumerSecret)  — raises RuntimeError nếu chưa cấu hình.

    Key ưu tiên:
      Encrypted : SSI_CONSUMER_ID_ENC / SSI_CONSUMER_SECRET_ENC
      Plaintext : SSI_CONSUMER_ID     / SSI_CONSUMER_SECRET
      Legacy    : SSI_ACCOUNT_ENC     / SSI_PASSWORD_ENC  (backward compat)

    Lấy consumerID/consumerSecret tại:
      https://iboard.ssi.com.vn/support/api-service/management
    """
    # ── Encrypted consumer keys (preferred) ───────────────────────────────────
    enc_id  = os.getenv("SSI_CONSUMER_ID_ENC", "")
    enc_sec = os.getenv("SSI_CONSUMER_SECRET_ENC", "")
    if enc_id and enc_sec:
        try:
            return decrypt_value(enc_id), decrypt_value(enc_sec)
        except Exception as exc:
            log.warning("secret_manager: giải mã consumer keys thất bại (%s)", exc)

    # ── Plaintext consumer keys ────────────────────────────────────────────────
    cid = os.getenv("SSI_CONSUMER_ID", "")
    sec = os.getenv("SSI_CONSUMER_SECRET", "")
    if cid and sec:
        return cid, sec

    # ── Legacy encrypted account/password (backward compat) ───────────────────
    enc_acc  = os.getenv("SSI_ACCOUNT_ENC", "")
    enc_pass = os.getenv("SSI_PASSWORD_ENC", "")
    if enc_acc and enc_pass:
        try:
            return decrypt_value(enc_acc), decrypt_value(enc_pass)
        except Exception as exc:
            log.warning("secret_manager: giải mã legacy keys thất bại (%s)", exc)

    # ── Legacy plaintext ───────────────────────────────────────────────────────
    acc  = os.getenv("SSI_ACCOUNT", "")
    pwd  = os.getenv("SSI_PASSWORD", "")
    if acc and pwd:
        return acc, pwd

    raise RuntimeError(
        "Chưa có SSI credentials.\n"
        "1. Đăng nhập iboard.ssi.com.vn → Support → API Service Management\n"
        "   Lấy consumerID và consumerSecret.\n"
        "2. Chạy: py -3.12 scripts/setup_ssi_credentials.py"
    )


def load_env(env_path: Path | None = None) -> None:
    """
    Đọc file .env và set os.environ (chỉ khi biến chưa tồn tại).
    Tự động gọi khi import analytics.ssi_iboard.
    """
    path = env_path or _ENV_PATH
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if key and key not in os.environ:
            os.environ[key] = val
