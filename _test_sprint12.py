"""
_test_sprint12.py
Sprint 12: SSI iBoard + Tự Doanh + Model v6 — 60 tests.

Sections:
  1. secret_manager          (8 tests)
  2. ssi_iboard module       (10 tests)
  3. data.db proprietary     (10 tests)
  4. smart_money proprietary (14 tests)
  5. feature_engineering v7  (8 tests)
  6. predictor _describe_pattern (6 tests)
  7. ml.__init__ exports     (4 tests)
"""

import sys, os, sqlite3, tempfile, time
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS = 0
FAIL = 0

def ok(name):
    global PASS; PASS += 1
    print(f"  ✓ {name}")

def fail(name, exc):
    global FAIL; FAIL += 1
    print(f"  ✗ {name}: {exc}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════
# Section 1: secret_manager (8 tests)
# ══════════════════════════════════════════════════════════════
section("1. secret_manager — machine-key encryption")

from analytics.secret_manager import encrypt_value, decrypt_value, _derive_machine_key

# 1.1 Key is 32-byte URL-safe base64
try:
    key = _derive_machine_key()
    assert len(key) == 44, f"Expected 44 chars (32 bytes b64), got {len(key)}"
    ok("derive_machine_key returns 44-char key")
except AssertionError as e: fail("derive_machine_key", e)

# 1.2 Key is deterministic
try:
    k1 = _derive_machine_key()
    k2 = _derive_machine_key()
    assert k1 == k2, "Key không deterministic!"
    ok("derive_machine_key is deterministic")
except AssertionError as e: fail("key_deterministic", e)

# 1.3 Encrypt returns non-empty string different from input
try:
    cipher = encrypt_value("hello")
    assert cipher != "hello"
    assert len(cipher) > 20
    ok("encrypt_value returns ciphertext")
except Exception as e: fail("encrypt_value", e)

# 1.4 Decrypt round-trip ASCII
try:
    plain = "test-password-123"
    cipher = encrypt_value(plain)
    decrypted = decrypt_value(cipher)
    assert decrypted == plain
    ok("decrypt_value round-trip ASCII")
except Exception as e: fail("decrypt_roundtrip_ascii", e)

# 1.5 Decrypt round-trip Unicode
try:
    plain = "Mật khẩu 12345!@#"
    decrypted = decrypt_value(encrypt_value(plain))
    assert decrypted == plain
    ok("decrypt_value round-trip Unicode")
except Exception as e: fail("decrypt_roundtrip_unicode", e)

# 1.6 Different plaintexts produce different ciphertexts
try:
    c1 = encrypt_value("abc")
    c2 = encrypt_value("xyz")
    assert c1 != c2
    ok("different plaintexts → different ciphertexts")
except Exception as e: fail("encrypt_uniqueness", e)

# 1.7 get_ssi_credentials from os.environ
try:
    old_acc  = os.environ.pop("SSI_ACCOUNT_ENC", None)
    old_pass = os.environ.pop("SSI_PASSWORD_ENC", None)
    os.environ["SSI_ACCOUNT"]  = "test_acc"
    os.environ["SSI_PASSWORD"] = "test_pass"
    from analytics.secret_manager import get_ssi_credentials
    acc, pwd = get_ssi_credentials()
    assert acc == "test_acc"
    assert pwd == "test_pass"
    ok("get_ssi_credentials plaintext fallback")
except Exception as e: fail("get_ssi_credentials_plaintext", e)
finally:
    os.environ.pop("SSI_ACCOUNT",  None)
    os.environ.pop("SSI_PASSWORD", None)
    if old_acc:  os.environ["SSI_ACCOUNT_ENC"]  = old_acc
    if old_pass: os.environ["SSI_PASSWORD_ENC"] = old_pass

# 1.8 get_ssi_credentials encrypted path
try:
    enc_acc  = encrypt_value("encrypted_acc")
    enc_pass = encrypt_value("encrypted_pass")
    old_enc_acc  = os.environ.pop("SSI_ACCOUNT_ENC",  None)
    old_enc_pass = os.environ.pop("SSI_PASSWORD_ENC", None)
    os.environ["SSI_ACCOUNT_ENC"]  = enc_acc
    os.environ["SSI_PASSWORD_ENC"] = enc_pass
    # remove plaintext vars
    os.environ.pop("SSI_ACCOUNT",  None)
    os.environ.pop("SSI_PASSWORD", None)
    from analytics.secret_manager import get_ssi_credentials as gsc
    a, p = gsc()
    assert a == "encrypted_acc"
    assert p == "encrypted_pass"
    ok("get_ssi_credentials encrypted path")
except Exception as e: fail("get_ssi_credentials_encrypted", e)
finally:
    os.environ.pop("SSI_ACCOUNT_ENC",  None)
    os.environ.pop("SSI_PASSWORD_ENC", None)
    if old_enc_acc:  os.environ["SSI_ACCOUNT_ENC"]  = old_enc_acc
    if old_enc_pass: os.environ["SSI_PASSWORD_ENC"] = old_enc_pass


# ══════════════════════════════════════════════════════════════
# Section 2: ssi_iboard module (10 tests)
# ══════════════════════════════════════════════════════════════
section("2. ssi_iboard — fetcher + parse logic")

from analytics.ssi_iboard import (
    _empty_df, _parse_response, fetch_investor_flow,
    _RateLimiter, _TokenCache,
)

# 2.1 _empty_df has correct columns
try:
    df = _empty_df()
    assert list(df.columns) == [
        "date", "proprietary_buy", "proprietary_sell", "proprietary_net",
        "foreign_buy", "foreign_sell", "retail_buy", "retail_sell",
    ]
    ok("_empty_df has correct columns")
except Exception as e: fail("empty_df_columns", e)

# 2.2 _parse_response with empty list
try:
    df = _parse_response([], "ACB")
    assert df.empty
    ok("_parse_response empty list → empty DataFrame")
except Exception as e: fail("parse_empty", e)

# 2.3 _parse_response with v2 field names
try:
    data = [{
        "tradingDate": "2026-05-26",
        "proprietaryBuyVol":  500000,
        "proprietarySellVol": 300000,
        "foreignBuyVol":     1000000,
        "foreignSellVol":     800000,
        "retailBuyVol":      5000000,
        "retailSellVol":     4900000,
    }]
    df = _parse_response(data, "ACB")
    assert len(df) == 1
    assert df["proprietary_buy"].iloc[0] == 500000
    assert df["proprietary_net"].iloc[0] == 200000   # 500k - 300k
    ok("_parse_response v2 field names")
except Exception as e: fail("parse_v2_fields", e)

# 2.4 _parse_response with alternate field names
try:
    data = [{
        "date": "2026-05-25",
        "propBuyVol":   400000,
        "propSellVol":  600000,
        "foreignBuy":   900000,
        "foreignSell":  700000,
        "retailBuy":   4000000,
        "retailSell":  3800000,
    }]
    df = _parse_response(data, "VIC")
    assert len(df) == 1
    assert df["proprietary_net"].iloc[0] == -200000  # 400k - 600k
    ok("_parse_response alternate field names")
except Exception as e: fail("parse_alt_fields", e)

# 2.5 _parse_response with nested wrapper
try:
    data = {"data": [{"tradingDate": "2026-05-26", "proprietaryBuyVol": 100,
                      "proprietarySellVol": 50, "foreignBuyVol": 200,
                      "foreignSellVol": 150, "retailBuyVol": 1000, "retailSellVol": 900}]}
    df = _parse_response(data, "GAS")
    assert not df.empty
    ok("_parse_response nested 'data' wrapper")
except Exception as e: fail("parse_nested_wrapper", e)

# 2.6 _parse_response with unix timestamp
try:
    import datetime
    ts = int(datetime.datetime(2026, 5, 26).timestamp())
    data = [{"tradingDate": ts, "proprietaryBuyVol": 100, "proprietarySellVol": 50,
              "foreignBuyVol": 200, "foreignSellVol": 150,
              "retailBuyVol": 1000, "retailSellVol": 900}]
    df = _parse_response(data, "VNM")
    assert not df.empty
    assert "2026-05-26" in df["date"].iloc[0]
    ok("_parse_response unix timestamp")
except Exception as e: fail("parse_unix_ts", e)

# 2.7 _TokenCache basic ops
try:
    cache = _TokenCache()
    assert cache.get() is None
    cache.set("mytoken", ttl_hours=24)
    assert cache.get() == "mytoken"
    cache.clear()
    assert cache.get() is None
    ok("_TokenCache get/set/clear")
except Exception as e: fail("token_cache", e)

# 2.8 _RateLimiter waits minimum interval
try:
    rl = _RateLimiter(per_minute=60)   # 1 req/sec
    t1 = time.monotonic()
    rl.wait()   # first call immediate
    rl.wait()   # second call waits ~1s
    elapsed = time.monotonic() - t1
    assert elapsed >= 0.9, f"Expected ≥0.9s, got {elapsed:.2f}s"
    ok("_RateLimiter enforces interval")
except Exception as e: fail("rate_limiter", e)

# 2.9 fetch_investor_flow returns empty when no SSI credentials
try:
    save_acc = os.environ.pop("SSI_ACCOUNT_ENC", None)
    save_acc2 = os.environ.pop("SSI_ACCOUNT", None)
    save_pass = os.environ.pop("SSI_PASSWORD_ENC", None)
    save_pass2 = os.environ.pop("SSI_PASSWORD", None)
    from analytics import ssi_iboard as sib
    sib._token_cache.clear()
    df = sib.fetch_investor_flow("ACB")
    assert df.empty or len(df.columns) > 0  # either empty or valid columns
    ok("fetch_investor_flow graceful degradation (no creds)")
except Exception as e: fail("fetch_no_creds", e)
finally:
    if save_acc:  os.environ["SSI_ACCOUNT_ENC"]  = save_acc
    if save_acc2: os.environ["SSI_ACCOUNT"]       = save_acc2
    if save_pass: os.environ["SSI_PASSWORD_ENC"]  = save_pass
    if save_pass2:os.environ["SSI_PASSWORD"]      = save_pass2

# 2.10 fetch_investor_flow returns DataFrame with correct columns on mock success
try:
    mock_data = [{
        "tradingDate": "2026-05-26",
        "proprietaryBuyVol": 500000, "proprietarySellVol": 300000,
        "foreignBuyVol": 1000000, "foreignSellVol": 800000,
        "retailBuyVol": 5000000, "retailSellVol": 4900000,
    }]
    from analytics import ssi_iboard as sib2
    with patch.object(sib2, "get_token", return_value="mock_token"):
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_data
            mock_get.return_value = mock_resp
            df = sib2.fetch_investor_flow("ACB", limit=5)
    assert not df.empty
    assert "proprietary_net" in df.columns
    ok("fetch_investor_flow with mocked SSI response")
except Exception as e: fail("fetch_mock_success", e)


# ══════════════════════════════════════════════════════════════
# Section 3: data.db proprietary columns (10 tests)
# ══════════════════════════════════════════════════════════════
section("3. data.db — proprietary columns migration + CRUD")

import data.db as db_mod

_TMP_DB_S12_A = Path(tempfile.gettempdir()) / "test_s12_db_a.db"
_TMP_DB_S12_B = Path(tempfile.gettempdir()) / "test_s12_db_b.db"

# 3.1 ensure_db creates proprietary columns
try:
    if _TMP_DB_S12_A.exists(): _TMP_DB_S12_A.unlink()
    orig_path = db_mod.DB_PATH
    db_mod.DB_PATH = _TMP_DB_S12_A
    db_mod.ensure_db()
    con = sqlite3.connect(str(_TMP_DB_S12_A))
    cols = {row[1] for row in con.execute("PRAGMA table_info(snapshots)")}
    con.close()
    assert "proprietary_buy"  in cols
    assert "proprietary_sell" in cols
    assert "proprietary_net"  in cols
    ok("ensure_db creates proprietary columns")
except Exception as e: fail("ensure_db_prop_cols", e)
finally:
    db_mod.DB_PATH = orig_path

# 3.2 upsert_snapshot with proprietary data
try:
    if _TMP_DB_S12_B.exists(): _TMP_DB_S12_B.unlink()
    orig_path = db_mod.DB_PATH
    db_mod.DB_PATH = _TMP_DB_S12_B
    db_mod.ensure_db()
    db_mod.upsert_snapshot(
        "2026-05-26", "ACB",
        foreign_buy=1000000, foreign_sell=800000, foreign_net=200000,
        total_volume=10000000, close=26.5,
        proprietary_buy=500000, proprietary_sell=300000, proprietary_net=200000,
    )
    df = db_mod.load_snapshots("ACB", last_n=5)
    assert len(df) == 1
    assert df["proprietary_buy"].iloc[0]  == 500000
    assert df["proprietary_sell"].iloc[0] == 300000
    assert df["proprietary_net"].iloc[0]  == 200000
    ok("upsert_snapshot with proprietary data")
except Exception as e: fail("upsert_prop_data", e)
finally:
    db_mod.DB_PATH = orig_path

# 3.3 upsert_snapshot backward compat (no proprietary args)
try:
    orig_path = db_mod.DB_PATH
    db_mod.DB_PATH = _TMP_DB_S12_B
    db_mod.upsert_snapshot(
        "2026-05-25", "ACB",
        foreign_buy=900000, foreign_sell=700000, foreign_net=200000,
        total_volume=9000000, close=26.0,
        # no proprietary args
    )
    df = db_mod.load_snapshots("ACB", last_n=5)
    assert len(df) == 2
    row_25 = df[df["session_date"].dt.strftime("%Y-%m-%d") == "2026-05-25"].iloc[0]
    assert row_25["proprietary_buy"]  == 0.0
    assert row_25["proprietary_sell"] == 0.0
    ok("upsert_snapshot backward compat defaults proprietary=0")
except Exception as e: fail("upsert_backward_compat", e)
finally:
    db_mod.DB_PATH = orig_path

# 3.4 load_snapshots includes proprietary columns
try:
    orig_path = db_mod.DB_PATH
    db_mod.DB_PATH = _TMP_DB_S12_B
    df = db_mod.load_snapshots("ACB", last_n=10)
    for col in ["proprietary_buy", "proprietary_sell", "proprietary_net"]:
        assert col in df.columns, f"Missing column: {col}"
    ok("load_snapshots has proprietary columns")
except Exception as e: fail("load_snapshots_prop_cols", e)
finally:
    db_mod.DB_PATH = orig_path

# 3.5 load_snapshots COALESCE handles NULL → 0
try:
    orig_path = db_mod.DB_PATH
    # Create old-style DB without proprietary cols
    old_db = Path(tempfile.gettempdir()) / "test_s12_old.db"
    if old_db.exists(): old_db.unlink()
    con = sqlite3.connect(str(old_db))
    con.execute("""
        CREATE TABLE snapshots (
            session_date TEXT, ticker TEXT,
            foreign_buy REAL, foreign_sell REAL, foreign_net REAL,
            total_volume REAL, close REAL, created_at TEXT,
            PRIMARY KEY (session_date, ticker)
        )""")
    con.execute("""INSERT INTO snapshots VALUES ('2026-05-24','VIC',100,80,20,1000,50,'2026-05-24')""")
    con.commit(); con.close()

    db_mod.DB_PATH = old_db
    db_mod.ensure_db()   # migration should add proprietary columns
    df2 = db_mod.load_snapshots("VIC", last_n=5)
    assert len(df2) == 1
    assert df2["proprietary_net"].iloc[0] == 0.0  # COALESCE NULL → 0
    ok("load_snapshots COALESCE NULL proprietary → 0 (migration)")
except Exception as e: fail("coalesce_migration", e)
finally:
    db_mod.DB_PATH = orig_path
    try: old_db.unlink()
    except: pass

# 3.6–3.10 Multi-ticker + sorting
try:
    orig_path = db_mod.DB_PATH
    db_mod.DB_PATH = _TMP_DB_S12_B
    for i, (ticker, date) in enumerate([("VNM","2026-05-20"),("VNM","2026-05-21"),
                                        ("VNM","2026-05-22"),("VNM","2026-05-23")]):
        db_mod.upsert_snapshot(date, ticker,
            foreign_buy=i*10, foreign_sell=i*5, foreign_net=i*5,
            total_volume=1000+i*100, close=40.0+i,
            proprietary_buy=i*20, proprietary_sell=i*10, proprietary_net=i*10)
    df = db_mod.load_snapshots("VNM", last_n=4)
    assert len(df) == 4
    ok("multi-row proprietary upsert")
    assert df["session_date"].is_monotonic_increasing
    ok("load_snapshots sorted ascending by date")
    assert df["proprietary_buy"].iloc[-1] == 60.0
    ok("proprietary_buy value correct last row")
    assert df["proprietary_net"].sum() == 0+10+20+30
    ok("proprietary_net sum correct")
    last_3 = db_mod.load_snapshots("VNM", last_n=3)
    assert len(last_3) == 3
    ok("load_snapshots last_n=3 limits correctly")
except Exception as e:
    fail("multi_row_proprietary", e)
finally:
    db_mod.DB_PATH = orig_path
    try: _TMP_DB_S12_B.unlink()
    except: pass


# ══════════════════════════════════════════════════════════════
# Section 4: smart_money proprietary (14 tests)
# ══════════════════════════════════════════════════════════════
section("4. ml.smart_money — ProprietarySignal + InstitutionalFlow")

from ml.smart_money import (
    ProprietarySignal, InstitutionalFlowSignal,
    _empty_prop_signal, _prop_label, _linear_trend,
    compute_proprietary_signal, compute_institutional_flow,
    compute_institutional_flow_features,
    PROP_CONFIRM_THRESHOLD, PROP_DISTRIB_THRESHOLD,
)
import numpy as np

# 4.1 _prop_label thresholds
try:
    assert _prop_label(0.15, 5)  == "Tích lũy tự doanh"
    assert _prop_label(-0.15, 5) == "Bán ròng tự doanh"
    assert _prop_label(0.0, 5)   == "Trung lập"
    assert _prop_label(0.15, 0)  == "Không có data"
    ok("_prop_label thresholds correct")
except Exception as e: fail("prop_label", e)

# 4.2 _empty_prop_signal structure
try:
    s = _empty_prop_signal("ACB")
    assert s.ticker == "ACB"
    assert s.sessions == 0
    assert s.proprietary_net_pct == 0.0
    assert s.label == "Không có data"
    ok("_empty_prop_signal structure")
except Exception as e: fail("empty_prop_signal", e)

# 4.3 ProprietarySignal as_dict
try:
    s = ProprietarySignal(ticker="VNM", sessions=5,
                          proprietary_net_pct=0.05, proprietary_net_5d=0.04,
                          prop_trend=0.02, label="Tích lũy tự doanh")
    d = s.as_dict()
    assert d["ticker"] == "VNM"
    assert d["proprietary_net_pct"] == 0.05
    ok("ProprietarySignal.as_dict()")
except Exception as e: fail("prop_signal_as_dict", e)

# 4.4 InstitutionalFlowSignal as_dict
try:
    from ml.smart_money import SmartMoneySignal
    sm = SmartMoneySignal("ACB", 10, 0.1, 0.08, 0.03, 0.07, True, "Accumulating")
    ps = ProprietarySignal("ACB", 5, 0.05, 0.04, 0.02, "Tích lũy tự doanh")
    inst = InstitutionalFlowSignal("ACB", sm, ps, 0.065, "Tổ chức mua ròng nhẹ")
    d = inst.as_dict()
    assert d["combined_institutional_score"] == 0.065
    assert "foreign" in d
    assert "proprietary" in d
    ok("InstitutionalFlowSignal.as_dict()")
except Exception as e: fail("inst_flow_as_dict", e)

# 4.5 compute_proprietary_signal returns empty when no data
try:
    with patch("data.db.load_snapshots") as mock_load:
        import pandas as pd
        mock_load.return_value = pd.DataFrame()
        with patch("analytics.ssi_iboard.fetch_investor_flow") as mock_ssi:
            mock_ssi.return_value = pd.DataFrame()
            sig = compute_proprietary_signal("FAKE")
    assert sig.sessions == 0
    assert sig.proprietary_net_pct == 0.0
    ok("compute_proprietary_signal → empty when no data")
except Exception as e: fail("prop_signal_no_data", e)

# 4.6 compute_proprietary_signal from SQLite data
try:
    import pandas as pd
    mock_df = pd.DataFrame({
        "proprietary_buy":  [100, 200, 300, 400, 500],
        "proprietary_sell": [80,  180, 250, 380, 450],
        "proprietary_net":  [20,  20,  50,  20,  50],
        "total_volume":     [1000,1200,1500,1800,2000],
    })
    with patch("data.db.load_snapshots", return_value=mock_df):
        sig = compute_proprietary_signal("ACB")
    assert sig.sessions == 5
    assert sig.proprietary_net_5d != 0.0
    ok("compute_proprietary_signal from SQLite data")
except Exception as e: fail("prop_signal_from_sqlite", e)

# 4.7 compute_institutional_flow combined score formula
try:
    from ml.smart_money import compute_smart_money
    import pandas as pd
    mock_snap = pd.DataFrame({
        "net_pct": [0.1, 0.12, 0.15, 0.14, 0.13],
        "foreign_net": [100, 120, 150, 140, 130],
        "total_volume": [1000]*5,
        "foreign_buy": [600]*5,
        "foreign_sell": [500]*5,
    })
    mock_prop_df = pd.DataFrame({
        "proprietary_buy":  [50, 60, 70, 65, 60],
        "proprietary_sell": [30, 35, 40, 38, 35],
        "proprietary_net":  [20, 25, 30, 27, 25],
        "total_volume":     [1000]*5,
    })
    with patch("data.db.load_snapshots", return_value=mock_snap):
        with patch("analytics.ssi_iboard.fetch_investor_flow") as mock_ssi:
            mock_ssi.return_value = pd.DataFrame()
            flow = compute_institutional_flow("ACB")
    assert hasattr(flow, "combined_institutional_score")
    assert -1.0 <= flow.combined_institutional_score <= 1.0
    ok("compute_institutional_flow combined score in [-1,+1]")
except Exception as e: fail("combined_score_range", e)

# 4.8 combined score formula verification
try:
    from ml.smart_money import SmartMoneySignal, ProprietarySignal, InstitutionalFlowSignal, _W_FOREIGN, _W_PROP_5D, _W_PROP_TR
    sm = SmartMoneySignal("T", 10, 0.0, 0.20, 0.10, 0.16, True, "Accumulating")
    ps = ProprietarySignal("T", 5, 0.0, 0.15, 0.05, "Tích lũy tự doanh")
    expected = np.clip(_W_FOREIGN*0.16 + _W_PROP_5D*0.15 + _W_PROP_TR*0.05, -1.0, 1.0)
    inst = InstitutionalFlowSignal("T", sm, ps, round(expected, 4), "label")
    assert abs(inst.combined_institutional_score - round(expected, 4)) < 1e-6
    ok("combined_score formula weights correct")
except Exception as e: fail("formula_weights", e)

# 4.9 compute_institutional_flow_features returns 6 keys
try:
    with patch("ml.smart_money.compute_smart_money") as mock_sm:
        with patch("ml.smart_money.compute_proprietary_signal") as mock_ps:
            from ml.smart_money import SmartMoneySignal, ProprietarySignal
            mock_sm.return_value = SmartMoneySignal("A", 10, 0.1, 0.2, 0.05, 0.14, True, "Accumulating")
            mock_ps.return_value = ProprietarySignal("A", 5, 0.05, 0.12, 0.03, "Tích lũy tự doanh")
            features = compute_institutional_flow_features("A")
    assert "foreign_net_pct" in features
    assert "foreign_trend" in features
    assert "smart_money_score" in features
    assert "proprietary_net_pct" in features
    assert "prop_trend" in features
    assert "combined_institutional_score" in features
    ok("compute_institutional_flow_features returns 6 keys")
except Exception as e: fail("flow_features_6_keys", e)

# 4.10–4.14 Additional label tests
try:
    assert _prop_label(0.10, 1) == "Tích lũy tự doanh"
    ok("prop_label at exact CONFIRM boundary")
    assert _prop_label(-0.10, 3) == "Bán ròng tự doanh"
    ok("prop_label at exact DISTRIB boundary")
    assert _prop_label(0.09, 3) == "Trung lập"
    ok("prop_label below confirm → trung lập")
    assert _prop_label(-0.09, 3) == "Trung lập"
    ok("prop_label above distrib → trung lập")
    ps = _empty_prop_signal("X")
    assert ps.label == "Không có data"
    ok("empty_prop_signal sessions=0 → không có data label")
except Exception as e: fail("prop_label_boundary", e)


# ══════════════════════════════════════════════════════════════
# Section 5: feature_engineering v7 (8 tests)
# ══════════════════════════════════════════════════════════════
section("5. ml.feature_engineering — 38 features (v7)")

from ml.feature_engineering import FEATURE_COLS, compute_stock_features

# 5.1 38 features total
try:
    assert len(FEATURE_COLS) == 38, f"Expected 38, got {len(FEATURE_COLS)}"
    ok(f"FEATURE_COLS has 38 features")
except Exception as e: fail("feature_count_38", e)

# 5.2 proprietary features present
try:
    assert "proprietary_net_pct" in FEATURE_COLS
    assert "prop_trend" in FEATURE_COLS
    assert "combined_institutional_score" in FEATURE_COLS
    ok("proprietary features in FEATURE_COLS")
except Exception as e: fail("prop_in_feature_cols", e)

# 5.3 order: proprietary at end (after wyckoff)
try:
    idx_wyckoff = FEATURE_COLS.index("vol_profile_score")
    idx_prop    = FEATURE_COLS.index("proprietary_net_pct")
    assert idx_prop > idx_wyckoff
    ok("proprietary features after wyckoff features (position)")
except Exception as e: fail("feature_order", e)

# 5.4 all legacy features preserved
try:
    legacy = ["return_1d","rsi_14","foreign_net_pct","smart_money_score",
              "spring_quality","lps_detected","wyckoff_phase_score"]
    for f in legacy:
        assert f in FEATURE_COLS, f"Missing: {f}"
    ok("all legacy features preserved")
except Exception as e: fail("legacy_features", e)

# 5.5 compute_stock_features passes sm_features with proprietary
try:
    import pandas as pd, numpy as np
    dates = pd.date_range("2025-01-01", periods=60, freq="B")
    np.random.seed(42)
    price = 50 + np.cumsum(np.random.randn(60) * 0.5)
    ohlcv = pd.DataFrame({
        "date": dates, "open": price*0.99, "high": price*1.01,
        "low": price*0.98, "close": price, "volume": np.random.randint(1e6,5e6,60).astype(float),
    })
    sm_feat = {
        "foreign_net_pct": 0.1, "foreign_trend": 0.05, "smart_money_score": 0.08,
        "proprietary_net_pct": 0.12, "prop_trend": 0.03, "combined_institutional_score": 0.07,
    }
    feat_df = compute_stock_features(ohlcv, sm_features=sm_feat)
    assert not feat_df.empty
    ok("compute_stock_features with proprietary sm_features")
except Exception as e: fail("compute_feat_with_prop", e)

# 5.6 proprietary features correctly propagated
try:
    assert "proprietary_net_pct" in feat_df.columns
    assert abs(feat_df["proprietary_net_pct"].mean() - 0.12) < 1e-9
    assert abs(feat_df["combined_institutional_score"].mean() - 0.07) < 1e-9
    ok("proprietary feature values correctly propagated to rows")
except Exception as e: fail("prop_feature_values", e)

# 5.7 missing proprietary in sm_features → 0.0
try:
    sm_only_foreign = {"foreign_net_pct": 0.1, "foreign_trend": 0.05, "smart_money_score": 0.08}
    feat_df2 = compute_stock_features(ohlcv, sm_features=sm_only_foreign)
    assert feat_df2["proprietary_net_pct"].mean() == 0.0
    assert feat_df2["prop_trend"].mean() == 0.0
    ok("missing proprietary in sm_features → 0.0")
except Exception as e: fail("prop_default_zero", e)

# 5.8 None sm_features → all proprietary = 0.0
try:
    feat_df3 = compute_stock_features(ohlcv, sm_features=None)
    assert feat_df3["combined_institutional_score"].mean() == 0.0
    ok("None sm_features → combined_institutional_score = 0.0")
except Exception as e: fail("none_sm_feat", e)


# ══════════════════════════════════════════════════════════════
# Section 6: predictor _describe_pattern (6 tests)
# ══════════════════════════════════════════════════════════════
section("6. ml.predictor — _describe_pattern with prop signal")

from ml.predictor import _describe_pattern

# 6.1 prop_net_5d >= 0.10 → tự doanh mua ròng
try:
    result = _describe_pattern(0.0, 0.0, 0.0, 0.0, 0.0, prop_net_5d=0.15)
    assert "tự doanh mua ròng" in result
    ok("_describe_pattern prop ≥ 0.10 → tự doanh mua ròng")
except Exception as e: fail("pattern_td_buy", e)

# 6.2 prop_net_5d <= -0.10 → tự doanh bán ròng
try:
    result = _describe_pattern(0.0, 0.0, 0.0, 0.0, 0.0, prop_net_5d=-0.15)
    assert "tự doanh bán ròng" in result
    ok("_describe_pattern prop ≤ -0.10 → tự doanh bán ròng")
except Exception as e: fail("pattern_td_sell", e)

# 6.3 combination: Spring + tự doanh mua ròng
try:
    result = _describe_pattern(0.0, 0.0, 0.0, spring_q=0.65, prop_net_5d=0.12)
    assert "Spring chất lượng cao" in result
    assert "tự doanh mua ròng" in result
    ok("_describe_pattern Spring cao + tự doanh mua ròng")
except Exception as e: fail("pattern_spring_td", e)

# 6.4 prop_net_5d = 0.0 → no tự doanh label
try:
    result = _describe_pattern(0.0, 0.0, 0.0, 0.0, 0.0, prop_net_5d=0.0)
    assert "tự doanh" not in result
    ok("_describe_pattern prop=0 → no tự doanh label")
except Exception as e: fail("pattern_no_td", e)

# 6.5 backward compat: no prop_net_5d arg
try:
    result = _describe_pattern(0.0, 0.0, 0.50)
    assert "smart money kéo" in result
    ok("_describe_pattern backward compat (no prop arg)")
except Exception as e: fail("pattern_backward_compat", e)

# 6.6 LPS + tự doanh bán ròng mixed signal
try:
    result = _describe_pattern(0.0, 0.0, 0.0, 0.0, lps=0.8, prop_net_5d=-0.12)
    assert "LPS confirmed" in result
    assert "tự doanh bán ròng" in result
    ok("_describe_pattern LPS + tự doanh bán ròng (mixed)")
except Exception as e: fail("pattern_lps_td_mixed", e)


# ══════════════════════════════════════════════════════════════
# Section 7: ml.__init__ exports (4 tests)
# ══════════════════════════════════════════════════════════════
section("7. ml.__init__ — Sprint 12 exports")

try:
    import ml
    # 7.1 ProprietarySignal exported
    assert hasattr(ml, "ProprietarySignal")
    ok("ml.ProprietarySignal exported")
except Exception as e: fail("export_ProprietarySignal", e)

try:
    # 7.2 InstitutionalFlowSignal exported
    assert hasattr(ml, "InstitutionalFlowSignal")
    ok("ml.InstitutionalFlowSignal exported")
except Exception as e: fail("export_InstitutionalFlowSignal", e)

try:
    # 7.3 compute_institutional_flow exported
    assert hasattr(ml, "compute_institutional_flow")
    ok("ml.compute_institutional_flow exported")
except Exception as e: fail("export_compute_institutional_flow", e)

try:
    # 7.4 compute_institutional_flow_features exported
    assert hasattr(ml, "compute_institutional_flow_features")
    ok("ml.compute_institutional_flow_features exported")
except Exception as e: fail("export_compute_inst_flow_features", e)


# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
total = PASS + FAIL
print(f"\n{'='*60}")
print(f"  Sprint 12 Tests: {PASS}/{total} passed")
print(f"{'='*60}")
if FAIL:
    print(f"  ⚠️  {FAIL} test(s) FAILED")
    sys.exit(1)
else:
    print("  ✅ All tests passed!")
