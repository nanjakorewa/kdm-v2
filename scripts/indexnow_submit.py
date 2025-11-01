#!/usr/bin/env python3
"""Submit site updates to the IndexNow API.

Usage examples:
  python scripts/indexnow_submit.py --url https://k-dm.work/foo/
  python scripts/indexnow_submit.py --sitemap public/sitemap.xml
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_BASE_URL = "https://k-dm.work/"
DEFAULT_ENDPOINT = "https://api.indexnow.org/indexnow"
DEFAULT_KEY_PATH = Path("static") / "d1b9a2bd02724df09ee54c2277dd1834.txt"


def load_key(path: Path) -> str:
  """Read the IndexNow site key from disk."""
  try:
    content = path.read_text(encoding="utf-8").strip()
  except OSError as exc:
    raise SystemExit(f"[indexnow] failed to read key file: {exc}") from exc
  if not content:
    raise SystemExit("[indexnow] key file is empty")
  return content


def parse_sitemap(path: Path) -> list[str]:
  """Extract URL list from a sitemap XML file."""
  try:
    tree = ET.parse(path)
  except (ET.ParseError, OSError) as exc:
    raise SystemExit(f"[indexnow] failed to parse sitemap: {exc}") from exc
  urls: list[str] = []
  root = tree.getroot()
  ns = ""
  if root.tag.startswith("{"):
    ns = root.tag.partition("}")[0] + "}"
  for loc in root.findall(f".//{ns}loc"):
    text = (loc.text or "").strip()
    if text:
      urls.append(text)
  if not urls:
    raise SystemExit(f"[indexnow] no <loc> entries found in {path}")
  return urls


def build_payload(host: str, key: str, key_location: str, urls: list[str]) -> bytes:
  """Create the JSON payload required by IndexNow."""
  data = {
    "host": host,
    "key": key,
    "keyLocation": key_location,
    "urlList": urls,
  }
  return json.dumps(data).encode("utf-8")


def submit(endpoint: str, payload: bytes, *, dry_run: bool) -> None:
  """Send the payload to the IndexNow endpoint."""
  if dry_run:
    print("[indexnow] dry-run (no request sent)")
    print(payload.decode("utf-8"))
    return
  request = urllib.request.Request(
    endpoint,
    data=payload,
    headers={"Content-Type": "application/json; charset=utf-8"},
    method="POST",
  )
  try:
    with urllib.request.urlopen(request) as response:
      body = response.read().decode("utf-8")
      status = response.status
  except OSError as exc:
    raise SystemExit(f"[indexnow] submission failed: {exc}") from exc
  print(f"[indexnow] submission OK (HTTP {status})")
  if body:
    print(body)


def main(argv: list[str]) -> int:
  parser = argparse.ArgumentParser(description="Ping IndexNow with updated URLs.")
  parser.add_argument(
    "--base-url",
    default=DEFAULT_BASE_URL,
    help="Published base URL of the site (default: %(default)s)",
  )
  parser.add_argument(
    "--endpoint",
    default=DEFAULT_ENDPOINT,
    help="IndexNow endpoint (default: %(default)s)",
  )
  parser.add_argument(
    "--key-file",
    type=Path,
    default=DEFAULT_KEY_PATH,
    help="Path to the IndexNow key file (default: %(default)s)",
  )
  parser.add_argument(
    "--sitemap",
    type=Path,
    help="Path to a sitemap.xml file to extract URLs from",
  )
  parser.add_argument(
    "--url",
    dest="urls",
    action="append",
    default=[],
    help="Submit a single URL (can be provided multiple times)",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Print the payload instead of sending it",
  )
  args = parser.parse_args(argv)

  base_url = args.base_url.rstrip("/") + "/"
  host = urllib.parse.urlparse(base_url).netloc
  if not host:
    raise SystemExit(f"[indexnow] base URL is invalid: {args.base_url}")

  urls = list(args.urls)
  if args.sitemap:
    urls.extend(parse_sitemap(args.sitemap))
  if not urls:
    raise SystemExit("[indexnow] at least one URL must be provided via --url or --sitemap")

  key_value = load_key(args.key_file)
  key_location = urllib.parse.urljoin(base_url, Path(args.key_file).name)
  payload = build_payload(host, key_value, key_location, urls)
  submit(args.endpoint, payload, dry_run=args.dry_run)
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
