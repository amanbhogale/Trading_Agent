# kite_auth.py
"""
One-time Kite Connect authentication helper.

Run this script once per trading day (access tokens expire at midnight).
It will:
  1. Print the login URL → open it in your browser
  2. Ask you to paste the request_token from the redirect URL
  3. Exchange it for an access_token
  4. Save both to .env (so dashboard.py picks them up automatically)

Usage:
    python kite_auth.py
"""

import os
import sys
import webbrowser
from pathlib import Path
from dotenv import load_dotenv, set_key

load_dotenv()

# ── Read credentials ──────────────────────────────────────────────────────
API_KEY    = os.getenv("KITE_API_KEY", "").strip()
API_SECRET = os.getenv("KITE_API_SECRET", "").strip()

if not API_KEY or not API_SECRET:
    print("\n❌  KITE_API_KEY and KITE_API_SECRET must be set in your .env file.")
    print("    See README for setup instructions.\n")
    sys.exit(1)

# ── Import kiteconnect ────────────────────────────────────────────────────
try:
    from kiteconnect import KiteConnect
except ImportError:
    print("❌  kiteconnect not installed. Run: pip install kiteconnect")
    sys.exit(1)

kite      = KiteConnect(api_key=API_KEY)
login_url = kite.login_url()

print("\n" + "=" * 60)
print("  KITE CONNECT — Daily Login")
print("=" * 60)
print(f"\n1. Opening login URL in your browser...")
print(f"   {login_url}\n")

try:
    webbrowser.open(login_url)
except Exception:
    print("   (Could not auto-open browser — copy the URL above manually)")

print("2. Log in with your Zerodha credentials + TOTP.")
print("3. After login, you'll be redirected to your redirect URL.")
print("   The URL will look like:")
print("   http://127.0.0.1:5000/?request_token=XXXXX&action=login&status=success")
print()

request_token = input("Paste the full redirect URL (or just the request_token): ").strip()

# Accept either the full URL or just the token
if "request_token=" in request_token:
    request_token = request_token.split("request_token=")[1].split("&")[0]

print(f"\n   Exchanging request_token for access_token...")

try:
    data         = kite.generate_session(request_token, api_secret=API_SECRET)
    access_token = data["access_token"]
    user_name    = data.get("user_name", "unknown")
    user_id      = data.get("user_id", "unknown")
except Exception as e:
    print(f"\n❌  Session generation failed: {e}")
    sys.exit(1)

# ── Persist to .env ───────────────────────────────────────────────────────
env_path = Path(".env")
if not env_path.exists():
    env_path.write_text("")

set_key(str(env_path), "KITE_ACCESS_TOKEN", access_token)

print(f"\n✅  Authenticated as: {user_name} ({user_id})")
print(f"   Access token saved to .env → KITE_ACCESS_TOKEN")
print(f"\n   You can now start the dashboard:")
print(f"   python dashboard.py\n")
print("=" * 60 + "\n")
