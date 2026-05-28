"""Debug: inspect raw FiinMarket GetForeign response fields."""
import sys, requests, json, time
sys.path.insert(0, ".")

url = "https://fiin-market.ssi.com.vn/MoneyFlow/GetForeign"
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Origin": "https://iboard.ssi.com.vn",
    "Referer": "https://iboard.ssi.com.vn/",
    "User-Agent": "Mozilla/5.0",
    "x-fiin-key": "KEY",
    "x-fiin-seed": "SEED",
    "x-fiin-user-id": "ID",
    "x-fiin-user-token": "0,212,108,244",
}
params = {"language": "vi", "ComGroupCode": "VN30", "time": int(time.time() * 1000)}
r = requests.get(url, params=params, headers=headers, timeout=15)
body = r.json()
item = body["items"][0]
today = item.get("today", {})

print("=== today keys ===")
print(list(today.keys()))
print()

for arr_name in ("buy", "sell", "netBuy", "netSell"):
    entries = today.get(arr_name) or []
    if entries:
        print(f"=== {arr_name}[0] keys ===")
        print(list(entries[0].keys()))
        acb = next((e for e in entries if str(e.get("ticker","")).upper() == "ACB"), None)
        target = acb or entries[0]
        print(json.dumps(target, indent=2, ensure_ascii=False))
        print()
        break
