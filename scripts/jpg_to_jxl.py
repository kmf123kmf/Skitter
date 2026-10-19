"""Batch-convert JPEG files to JPEG XL with zero loss.

JPEG XL can repack a JPEG's compressed data (no decode, no re-encode), and
keeps what it needs to rebuild the original file byte for byte. Every file is
checked: the .jxl is decoded back to a JPEG and compared with the original,
and is only written if they match exactly. Files typically shrink by 10-20%.

Needs: pip install pillow pillow-jxl-plugin

    python jpg_to_jxl.py PHOTOS                  # .jxl next to each .jpg
    python jpg_to_jxl.py PHOTOS -o OUT -r        # mirror the tree into OUT
    python jpg_to_jxl.py PHOTOS --delete-originals

To get a JPEG back: djxl photo.jxl photo.jpg (libjxl), or decode it with
pillow_jxl.Decoder, whose third return value is the original JPEG's bytes.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from pillow_jxl import Decoder, Encoder

JPEG_SUFFIXES = {".jpg", ".jpeg", ".jpe", ".jfif"}


def convert(src: Path, dst: Path, effort: int, overwrite: bool) -> tuple[str, int, int]:
    """Converts one file; returns (status, original size, new size)."""
    if dst.exists() and not overwrite:
        return "skipped (exists)", 0, 0
    raw = src.read_bytes()
    with Image.open(src) as im:
        if im.format != "JPEG":
            return f"skipped (not a JPEG: {im.format})", 0, 0
        mode, width, height = im.mode, im.width, im.height
    encoder = Encoder(
        mode=mode,
        lossless=False,
        quality=100,
        decoding_speed=0,
        effort=effort,
        use_container=True,
        use_original_profile=False,
        num_threads=1,
    )
    jxl = encoder(raw, width, height, jpeg_encode=True)
    is_jpeg, _info, rebuilt, *_ = Decoder(num_threads=1)(jxl)
    if not is_jpeg or rebuilt != raw:
        return "FAILED (round trip not bit-exact; nothing written)", len(raw), 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    tmp.write_bytes(jxl)
    stat = src.stat()
    os.utime(tmp, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    tmp.replace(dst)
    return "ok", len(raw), len(jxl)


def find_jpegs(root: Path, recursive: bool) -> list[Path]:
    if root.is_file():
        return [root]
    pattern = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in pattern if p.is_file() and p.suffix.lower() in JPEG_SUFFIXES)


def target(src: Path, root: Path, out: Path | None) -> Path:
    if out is None:
        return src.with_suffix(".jxl")
    relative = src.relative_to(root) if root.is_dir() else Path(src.name)
    return (out / relative).with_suffix(".jxl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="a JPEG file or a folder of them")
    parser.add_argument(
        "-o", "--output", type=Path, help="output folder (default: next to each JPEG)"
    )
    parser.add_argument("-r", "--recursive", action="store_true", help="include subfolders")
    parser.add_argument(
        "-e",
        "--effort",
        type=int,
        default=7,
        choices=range(1, 10),
        metavar="1-9",
        help="encoder effort: higher is smaller and slower (default 7)",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=os.cpu_count() or 1, help="parallel workers"
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing .jxl files")
    parser.add_argument(
        "--delete-originals",
        action="store_true",
        help="delete each JPEG once its .jxl is written and verified",
    )
    args = parser.parse_args()

    root = args.input.resolve()
    if not root.exists():
        parser.error(f"{root} does not exist")
    out = args.output.resolve() if args.output else None
    sources = find_jpegs(root, args.recursive)
    if not sources:
        print("No JPEG files found.")
        return 0

    before = after = converted = failed = 0
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {
            pool.submit(convert, src, target(src, root, out), args.effort, args.overwrite): src
            for src in sources
        }
        for done, future in enumerate(as_completed(futures), 1):
            src = futures[future]
            try:
                status, old, new = future.result()
            except Exception as error:  # a corrupt or unsupported JPEG (e.g. some CMYK files)
                status, old, new = f"FAILED ({type(error).__name__}: {error})", 0, 0
            if status == "ok":
                converted += 1
                before += old
                after += new
                if args.delete_originals:
                    src.unlink()
                detail = f"{old:,} -> {new:,} bytes ({new / old:.0%})"
            else:
                failed += status.startswith("FAILED")
                detail = status
            print(f"[{done}/{len(sources)}] {src}: {detail}", flush=True)

    print(f"\nConverted {converted} of {len(sources)} files", end="")
    if converted:
        print(f": {before:,} -> {after:,} bytes, saved {1 - after / before:.1%}", end="")
    print(f"; {failed} failed." if failed else ".")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
