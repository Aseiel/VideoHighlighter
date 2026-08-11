"""Generate assets/splash.png — the image PyInstaller's bootloader shows while
the frozen app starts.

Committed as a tool rather than done by hand so the splash can be regenerated
from assets/icon.png whenever the artwork changes. The taskbar icon once drifted
out of sync with the logo for exactly this reason (see the icon-generation note
in main.py), and a splash is even easier to forget.

    python tools/make_splash.py

Constraints that shaped it, all from PyInstaller's Splash:
  - PNG only, and no larger than 760x480 (bigger images need Pillow at *build*
    time to resize, which CI should not have to depend on).
  - Magenta #ff00ff is the Windows transparency key, so it must not appear.
  - The bootloader can overlay progress text, but only when the .spec sets
    `text_pos`; the `--splash` CLI flag hardcodes it to None. The build uses the
    CLI flag, so this image carries no space reserved for text and has to read
    as finished on its own — the app's own Qt splash takes over the moment Qt
    is up (modules/startup_splash.py).
"""

from __future__ import annotations

import os
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "assets", "icon.png")
TARGET = os.path.join(ROOT, "assets", "splash.png")

W, H = 480, 300
# The artwork is not flat: it runs from a cool near-black at the top to a
# darker one at the bottom. The canvas matches that gradient and the logo's
# edges are feathered into it, so the square it is cropped to disappears
# instead of sitting on the splash as a visible tile.
BG_TOP = (23, 24, 28)
BG_BOTTOM = (15, 16, 18)
BORDER = (61, 61, 61)          # THEME.border_strong
TEXT = (230, 230, 230)         # THEME.text
ACCENT = (238, 232, 22)        # the logo's yellow brush stroke
LOGO_PX = 150
LOGO_TOP = 36
FEATHER = 10                   # px of alpha falloff around the logo

# Edition-neutral on purpose: the same file then serves both editions, and the
# running app puts the real edition and version on its own splash anyway.
WORDMARK = "VideoHighlighter"


def _font(size: int):
    """A real UI font where one exists, Pillow's bitmap default otherwise —
    ugly, but never a crash on a machine without the font."""
    for name in ("segoeui.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        for base in (r"C:\Windows\Fonts", "/usr/share/fonts/truetype/dejavu",
                     "/Library/Fonts"):
            path = os.path.join(base, name)
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    pass
    return ImageFont.load_default()


def _background() -> Image.Image:
    img = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img)
    for y in range(H):
        t = y / max(1, H - 1)
        draw.line([(0, y), (W, y)], fill=tuple(
            round(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM)))
    return img


def _feathered_logo() -> tuple[Image.Image, Image.Image]:
    logo = Image.open(SOURCE).convert("RGB").resize((LOGO_PX, LOGO_PX),
                                                    Image.LANCZOS)
    mask = Image.new("L", (LOGO_PX, LOGO_PX), 0)
    ImageDraw.Draw(mask).rectangle(
        [FEATHER, FEATHER, LOGO_PX - 1 - FEATHER, LOGO_PX - 1 - FEATHER],
        fill=255)
    return logo, mask.filter(ImageFilter.GaussianBlur(FEATHER / 2))


def build() -> Image.Image:
    img = _background()
    logo, mask = _feathered_logo()
    img.paste(logo, ((W - LOGO_PX) // 2, LOGO_TOP), mask)

    draw = ImageDraw.Draw(img)
    font = _font(21)
    box = draw.textbbox((0, 0), WORDMARK, font=font)
    draw.text(((W - (box[2] - box[0])) // 2, 210), WORDMARK,
              font=font, fill=TEXT)

    # A short accent rule, echoing the brush stroke under the V.
    draw.rectangle([W // 2 - 34, 250, W // 2 + 34, 252], fill=ACCENT)

    # Border last, so nothing pasted above can cover it.
    draw.rectangle([0, 0, W - 1, H - 1], outline=BORDER, width=1)
    return img


def main() -> int:
    if not os.path.exists(SOURCE):
        print(f"❌ Missing source artwork: {SOURCE}")
        return 1
    img = build()

    # Guard the transparency key rather than trusting the palette above: a
    # future tweak that reintroduces magenta would punch a hole in the splash
    # on Windows, and that is not obvious from looking at the code.
    if any(c == (255, 0, 255) for _n, c in (img.getcolors(W * H) or [])):
        print("❌ Image contains #ff00ff, which Windows treats as transparent")
        return 1

    img.save(TARGET, "PNG")
    print(f"✅ Wrote {TARGET} ({img.width}x{img.height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
