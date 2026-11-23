#!/usr/bin/env python3
"""
make_icon.py
─────────────
Generates AppIcon.icns for LocalTranscription.app using Pillow.
Run via create_app_bundle.command — requires macOS (uses iconutil).

Usage:
    python make_icon.py /path/to/LocalTranscription.app
"""

import sys
import subprocess
import shutil
from pathlib import Path


def make_icon(app_path: Path):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("  ⚠ Pillow not found — skipping icon generation")
        return

    resources = app_path / "Contents" / "Resources"
    resources.mkdir(parents=True, exist_ok=True)
    iconset = resources / "AppIcon.iconset"
    iconset.mkdir(exist_ok=True)

    # ── Draw 1024×1024 master image ───────────────────────────────────────────
    size = 1024
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background — dark navy circle
    draw.ellipse([0, 0, size - 1, size - 1], fill=(26, 26, 46, 255))

    # Slight inner glow ring
    draw.ellipse([20, 20, size - 21, size - 21],
                 outline=(83, 192, 240, 40), width=18)

    cx, cy = size // 2, size // 2

    # ── Microphone body (pill) ─────────────────────────────────────────────────
    mw, mh = 210, 320
    mx0 = cx - mw // 2
    my0 = cy - mh // 2 - 80
    draw.rounded_rectangle(
        [mx0, my0, mx0 + mw, my0 + mh],
        radius=mw // 2,
        fill=(83, 192, 240, 255),
    )
    # Inner highlight stripe
    draw.rounded_rectangle(
        [mx0 + 30, my0 + 30, mx0 + mw - 60, my0 + mh // 2],
        radius=40,
        fill=(255, 255, 255, 30),
    )

    # ── Sound-wave arcs ────────────────────────────────────────────────────────
    mic_mid_y = my0 + mh // 2
    for r in [120, 180]:
        alpha = 160 if r == 120 else 90
        # Left arc
        draw.arc(
            [cx - r, mic_mid_y - r, cx + r, mic_mid_y + r],
            start=210, end=270,
            fill=(83, 192, 240, alpha), width=20,
        )
        # Right arc
        draw.arc(
            [cx - r, mic_mid_y - r, cx + r, mic_mid_y + r],
            start=270, end=330,
            fill=(83, 192, 240, alpha), width=20,
        )

    # ── Stand arc ─────────────────────────────────────────────────────────────
    stand_w, stand_h = 360, 190
    stand_y = my0 + mh - 50
    draw.arc(
        [cx - stand_w // 2, stand_y, cx + stand_w // 2, stand_y + stand_h],
        start=0, end=180,
        fill=(83, 192, 240, 255),
        width=28,
    )

    # ── Pole ──────────────────────────────────────────────────────────────────
    pole_top = stand_y + stand_h // 2
    draw.rounded_rectangle(
        [cx - 14, pole_top, cx + 14, pole_top + 85],
        radius=7,
        fill=(83, 192, 240, 255),
    )

    # ── Base ──────────────────────────────────────────────────────────────────
    draw.rounded_rectangle(
        [cx - 90, pole_top + 72, cx + 90, pole_top + 105],
        radius=18,
        fill=(83, 192, 240, 255),
    )

    # ── Export at all required iconset sizes ───────────────────────────────────
    for s in [16, 32, 64, 128, 256, 512]:
        img.resize((s, s), Image.LANCZOS).save(iconset / f"icon_{s}x{s}.png")
        img.resize((s * 2, s * 2), Image.LANCZOS).save(iconset / f"icon_{s}x{s}@2x.png")
    # 512@2x = 1024
    img.save(iconset / "icon_512x512@2x.png")

    # ── Convert with iconutil ──────────────────────────────────────────────────
    icns_path = resources / "AppIcon.icns"
    result = subprocess.run(
        ["iconutil", "-c", "icns", str(iconset), "-o", str(icns_path)],
        capture_output=True,
    )
    if result.returncode == 0:
        shutil.rmtree(iconset)
        print(f"  ✓ Icon saved → {icns_path}")
    else:
        print(f"  ⚠ iconutil error: {result.stderr.decode().strip()}")
        print(f"    (iconset left at {iconset} — you can convert manually)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python make_icon.py /path/to/LocalTranscription.app")
        sys.exit(1)
    make_icon(Path(sys.argv[1]))
