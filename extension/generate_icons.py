"""
DeepGuard — Icon Generator
Generates all required toolbar icons in 4 states × 4 sizes using Pillow.
Run: python generate_icons.py
"""

from PIL import Image, ImageDraw
import os

SIZES = [16, 32, 48, 128]

COLORS = {
    'green': ('#27A96C', '#1a7a4e'),
    'amber': ('#E89E1A', '#a87010'),
    'red':   ('#D63939', '#9a2626'),
    'gray':  ('#666666', '#444444'),
}

OUTPUT_DIR = 'icons'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def make_icon(name, fill_hex, stroke_hex, size):
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    pad = max(1, size // 10)
    r   = (size // 2) - pad

    cx, cy = size // 2, size // 2

    # Outer circle
    draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        fill=fill_hex,
        outline=stroke_hex,
        width=max(1, size // 20)
    )

    # Inner dot (shield / check feel)
    inner = r // 3
    draw.ellipse(
        [cx - inner, cy - inner, cx + inner, cy + inner],
        fill='white'
    )

    return img

for color_name, (fill, stroke) in COLORS.items():
    for size in SIZES:
        icon = make_icon(color_name, fill, stroke, size)
        path = os.path.join(OUTPUT_DIR, f'icon-{color_name}-{size}.png')
        icon.save(path)
        print(f'  Saved {path}')

print('\nAll icons generated.')
