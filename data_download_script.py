"""
download_data.py
================
PhysioNet gaitpdb downloader — corrected URL and filename format.

Real URL  : https://physionet.org/protected/published-projects/gaitpdb/1.0.0/
Filenames : GaCo01_01.txt, GaPd01_01.txt, JuCo01_01.txt, etc.
            ^^                             ^^
            Ga=dataset  Co=Control         Ju=dataset  Pd=Parkinson's

Sorted locally into:
    data/Ga/   all files starting with Ga
    data/Ju/   all files starting with Ju
    data/Si/   all files starting with Si

Run: python download_data.py
"""

import re
import sys
import requests
from pathlib import Path

BASE     = "https://physionet.org/files/gaitpdb/1.0.0/"
OUT_DIR  = Path("./data")
USERNAME = "vt004"
PASSWORD = "Vaibhav@1234"

session = requests.Session()
session.auth = (USERNAME, PASSWORD)

print("=" * 55)
print("  PhysioNet gaitpdb downloader")
print("=" * 55)
print(f"\n  Fetching: {BASE}")

r = session.get(BASE, timeout=15)
print(f"  Status  : {r.status_code}")

if r.status_code == 401:
    print("  ERROR: bad credentials.")
    sys.exit(1)
if r.status_code == 403:
    print("  ERROR: access denied — sign the data use agreement at:")
    print("  https://physionet.org/content/gaitpdb/1.0.0/")
    sys.exit(1)
if r.status_code != 200:
    print(f"  ERROR: unexpected status {r.status_code}")
    print(r.text[:400])
    sys.exit(1)

# Parse all .txt hrefs
all_hrefs = re.findall(r'href=["\']([^"\']+)["\']', r.text)
txt_files = list(dict.fromkeys(
    h.split("/")[-1]
    for h in all_hrefs
    if h.lower().endswith(".txt")
))
print(f"  Files found: {len(txt_files)}")

if not txt_files:
    print("\n  No .txt files found. Page preview:")
    print(r.text[:1000])
    sys.exit(1)

# Sort into Ga / Ju / Si buckets
prefix_map = {"Ga": [], "Ju": [], "Si": []}
skipped    = []
for fname in txt_files:
    matched = False
    for prefix in prefix_map:
        if fname.startswith(prefix):
            prefix_map[prefix].append(fname)
            matched = True
            break
    if not matched:
        skipped.append(fname)

for prefix, files in prefix_map.items():
    print(f"  {prefix}: {len(files)} files  "
          f"(e.g. {files[0] if files else 'none'})")
    (OUT_DIR / prefix).mkdir(parents=True, exist_ok=True)

if skipped:
    print(f"  Skipped (no matching prefix): {skipped[:5]}")

# Download
total = 0
for prefix, files in prefix_map.items():
    print(f"\n  Downloading {prefix} ({len(files)} files) ...")
    for fname in files:
        fpath = OUT_DIR / prefix / fname
        if fpath.exists() and fpath.stat().st_size > 0:
            print(f"    [skip] {fname}")
            continue

        resp = session.get(BASE + fname, timeout=30)
        if resp.status_code != 200:
            print(f"    [ERROR {resp.status_code}] {fname}")
            continue
        if b"<!DOCTYPE" in resp.content[:100]:
            print(f"    [ERROR] Got HTML for {fname} — auth issue")
            continue

        fpath.write_bytes(resp.content)
        print(f"    [OK] {fname}  ({len(resp.content)/1024:.1f} KB)")
        total += 1

print(f"\n  Total downloaded: {total}")
print("  Done.")
