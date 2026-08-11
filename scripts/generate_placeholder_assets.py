"""Regenerate the original, copyright-safe placeholder PNG assets."""

from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 512
OUT = Path(__file__).resolve().parents[1] / "assets"


def static_logo() -> Image.Image:
    image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    cyan = (58, 238, 225, 235)
    dark = (7, 18, 29, 235)
    draw.ellipse((116, 116, 396, 396), fill=dark, outline=cyan, width=12)
    draw.polygon([(256, 155), (340, 256), (256, 357), (172, 256)], outline=cyan, width=14)
    draw.line((188, 256, 324, 256), fill=cyan, width=10)
    draw.line((256, 188, 256, 324), fill=cyan, width=10)
    return image


def ring_logo() -> Image.Image:
    image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    cyan = (58, 238, 225, 230)
    draw.arc((36, 36, 476, 476), 8, 114, fill=cyan, width=18)
    draw.arc((36, 36, 476, 476), 138, 246, fill=cyan, width=18)
    draw.arc((36, 36, 476, 476), 270, 350, fill=cyan, width=18)
    for angle_box in [(72, 72, 440, 440), (88, 88, 424, 424)]:
        draw.arc(angle_box, 196, 330, fill=(101, 145, 255, 190), width=5)
    return image


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    static_logo().save(OUT / "placeholder_static.png")
    ring_logo().save(OUT / "placeholder_ring.png")


if __name__ == "__main__":
    main()
