"""Download the FY26 annual reports from each company's own website into data/raw/.

Usage: python scripts/fetch_reports.py [--force]

Files already present (and valid PDFs) are skipped. Some sites block scripted
downloads (Infosys returns 403 to non-browser clients); for those the script
prints the page to download from by hand into data/raw/ under the expected name.
"""
import argparse
import sys
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

REPORTS = [
    {
        "company": "TCS",
        "filename": "tcs_ar_fy26.pdf",
        "url": "https://www.ar.tcs.com/files/TCS_Annual_Report_2025-26.pdf",
        "page": "https://www.ar.tcs.com/",
    },
    {
        "company": "Infosys",
        "filename": "infosys_ar_fy26.pdf",
        "url": "https://www.infosys.com/investors/reports-filings/annual-report/annual/documents/infosys-ar-26.pdf",
        "page": "https://www.infosys.com/investors/reports-filings/annual-report/annual-reports/ar-2025-26.html",
    },
    {
        "company": "Wipro",
        "filename": "wipro_ar_fy26.pdf",
        "url": "https://www.wipro.com/content/dam/nexus/en/investor/annual-reports/2025-2026/Integrated-annual-report-2025-26.pdf",
        "page": "https://www.wipro.com/investors/annual-reports/",
    },
]


def is_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".part")
    with requests.get(url, headers=HEADERS, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with tmp.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    if not is_pdf(tmp):
        tmp.unlink()
        raise ValueError("response was not a PDF")
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for r in REPORTS:
        dest = RAW_DIR / r["filename"]
        if dest.exists() and is_pdf(dest) and not args.force:
            print(f"[skip] {r['company']}: {dest.name} already present")
            continue
        print(f"[get ] {r['company']}: {r['url']}")
        try:
            download(r["url"], dest)
            print(f"[ ok ] saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        except (requests.RequestException, ValueError) as e:
            print(f"[FAIL] {r['company']}: {e}")
            failed.append(r)

    if failed:
        print("\nDownload these by hand in a browser and save into data/raw/:")
        for r in failed:
            print(f"  {r['company']}: open {r['page']}\n    save as {RAW_DIR / r['filename']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
