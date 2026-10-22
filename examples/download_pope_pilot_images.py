# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dongeunk (original project contributions).
# Licensed under the Apache License, Version 2.0 (the License).
# You may not use this file except in compliance with the License.
# A copy is in the repository LICENSE and at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by law or agreed to in writing, distributed AS IS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See the License for permissions and limitations.
# Publication copy: license/attribution comments added; executable code unchanged.

"""Download only the 30 original COCO JPEGs in the existing POPE manifest.

Does not change questions, annotations, existing evaluation code, or pixels.
COCO's official download source lists gs://images.cocodataset.org/val2014;
the fallback below uses that exact bucket through Google's public XML endpoint.
No archive/model/package downloads are implemented.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import time
import urllib.request

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/pope_pilot/sample_manifest.jsonl"
PREFLIGHT = ROOT / "results/pope_pilot/preflight.json"
IMAGE_DIR = (ROOT / "data/coco/val2014").resolve()
REPORT = ROOT / "results/pope_pilot/image_downloads.json"
BASE_URLS = (
    "http://images.cocodataset.org/val2014/",
    "https://images.cocodataset.org/val2014/",
    "https://storage.googleapis.com/images.cocodataset.org/val2014/",
)
MAX_BYTES = 16 * 1024 * 1024


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_jpeg(data):
    if not data.startswith(b"\xff\xd8\xff"):
        raise ValueError("Response is not a JPEG by magic bytes")
    with Image.open(BytesIO(data)) as source:
        if source.format != "JPEG":
            raise ValueError(f"Expected JPEG, got {source.format}")
        size, mode = source.size, source.mode
        source.verify()
    with Image.open(BytesIO(data)) as source:
        source.load()  # Decode every pixel; verify() alone does not decode JPEG.
    return dict(format="JPEG", width=size[0], height=size[1], mode=mode,
                bytes=len(data), sha256=digest(data), pil_verify=True, pil_decode=True)


def download_one(record):
    name = record["image"]
    target = (IMAGE_DIR / name).resolve()
    if target.parent != IMAGE_DIR or target != Path(record["image_path"]).resolve():
        raise ValueError(f"Manifest destination outside expected directory: {name}")
    attempts = []
    if target.is_file():
        try:
            return dict(image=name, image_id=record["image_id"], path=str(target),
                        downloaded=False, reused=True, attempts=[], **validate_jpeg(target.read_bytes()))
        except Exception as error:
            attempts.append(dict(url=None, error=f"Existing image invalid: {type(error).__name__}: {error}"))
    for retry in range(2):
        for base in BASE_URLS:
            url = base + name
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "POPE-Pilot/1.0"})
                with urllib.request.urlopen(request, timeout=20) as response:
                    content_type = response.headers.get("Content-Type")
                    declared = response.headers.get("Content-Length")
                    if declared is not None and int(declared) > MAX_BYTES:
                        raise ValueError("Response exceeds per-JPEG byte limit")
                    data = response.read(MAX_BYTES + 1)
                    if len(data) > MAX_BYTES:
                        raise ValueError("Response exceeds per-JPEG byte limit")
                    if declared is not None and len(data) != int(declared):
                        raise ValueError("Incomplete HTTP response")
                    final_url = response.geturl()
                metadata = validate_jpeg(data)
                temporary = target.with_suffix(".jpg.part")
                if temporary.resolve().parent != IMAGE_DIR or target.parent != IMAGE_DIR:
                    raise ValueError("Unsafe atomic image destination")
                temporary.write_bytes(data)
                if digest(temporary.read_bytes()) != metadata["sha256"]:
                    raise ValueError("Written bytes differ from HTTP response")
                temporary.replace(target)
                disk_metadata = validate_jpeg(target.read_bytes())
                if disk_metadata != metadata:
                    raise ValueError("Stored JPEG validation differs from response")
                attempts.append(dict(url=url, status="ok", retry=retry))
                return dict(image=name, image_id=record["image_id"], path=str(target),
                            downloaded=True, reused=False, source_url=url, final_url=final_url,
                            content_type=content_type, attempts=attempts, **metadata)
            except Exception as error:
                attempts.append(dict(url=url, retry=retry, error=f"{type(error).__name__}: {error}"))
        time.sleep(1)
    return dict(image=name, image_id=record["image_id"], path=str(target),
                downloaded=False, reused=False, error="All official individual URLs failed", attempts=attempts)


def main():
    before = {str(path): digest(path.read_bytes()) for path in (MANIFEST, PREFLIGHT)}
    records = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = json.loads(PREFLIGHT.read_text(encoding="utf-8"))["missing_images"]
    if len(records) != 30 or len({r["image"] for r in records}) != 30:
        raise ValueError("Expected exactly 30 distinct fixed-manifest images")
    if {r["image"] for r in records} != {r["image"] for r in missing}:
        raise ValueError("Manifest and existing preflight image sets differ")
    if sum(r["label"] == "yes" for r in records) != 15 or sum(r["label"] == "no" for r in records) != 15:
        raise ValueError("Fixed manifest must retain the original 15/15 balance")
    for row in records:
        match = re.fullmatch(r"COCO_val2014_(\d{12})\.jpg", row["image"])
        if match is None or int(match.group(1)) != row["image_id"]:
            raise ValueError("Manifest image filename/ID mismatch")
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download_one, record): record for record in records}
        for future in as_completed(futures):
            record = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = dict(image=record["image"], error=f"{type(error).__name__}: {error}")
            results[record["image"]] = result
            print(f"[{len(results)}/30] {record['image']}: "
                  f"{result.get('error', str(result.get('bytes')) + ' bytes; JPEG OK')}", flush=True)
    unchanged = all(digest(Path(path).read_bytes()) == value for path, value in before.items())
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  manifest_sha256=before[str(MANIFEST)], preflight_sha256=before[str(PREFLIGHT)],
                  fixed_manifest_unchanged=unchanged, expected_images=30,
                  downloaded_count=sum(r.get("downloaded", False) for r in results.values()),
                  validated_count=sum(r.get("pil_decode", False) for r in results.values()),
                  total_bytes=sum(r.get("bytes", 0) for r in results.values()),
                  full_archive_downloaded=False, pixels_modified=False,
                  source_documentation="https://raw.githubusercontent.com/cocodataset/cocodataset.github.io/master/dataset/download.htm",
                  seconds=time.perf_counter() - start,
                  files=[results[record["image"]] for record in records])
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2), flush=True)
    return 0 if unchanged and report["validated_count"] == 30 else 1


if __name__ == "__main__":
    raise SystemExit(main())
