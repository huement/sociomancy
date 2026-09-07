import os
import json
import asyncio
import argparse
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from pyppeteer import launch
from PIL import Image, ImageChops
try:
    from moviepy import ImageSequenceClip
except ImportError:
    from moviepy.editor import ImageSequenceClip

# Video Render Controls
DURATION = 3  # Seconds to record ECharts animation
FPS = 25  # Frames per second
VIEWPORT_WIDTH = 1600  # Video Canvas Width
VIEWPORT_HEIGHT = 1200  # Video Canvas Height
TEMP_FRAME_DIR = Path("data/temp_chart_frames")

CHARTS_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = CHARTS_DIR / "templates"


def load_theme() -> str:
    """Loads custom ECharts dark theme JSON string."""
    theme_json_path = CHARTS_DIR / "chart-theme.json"
    if theme_json_path.exists():
        with open(theme_json_path, "r", encoding="utf-8") as f:
            return f.read()
    return "{}"


def autocrop_frame(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    return im.crop(bbox) if bbox else im


async def record_html_to_frames(html_path: Path):
    if TEMP_FRAME_DIR.exists():
        for f in TEMP_FRAME_DIR.glob("*.png"):
            f.unlink()
    TEMP_FRAME_DIR.mkdir(parents=True, exist_ok=True)

    browser = await launch(headless=True, args=["--no-sandbox"])
    page = await browser.newPage()
    await page.setViewport({
        "width": VIEWPORT_WIDTH,
        "height": VIEWPORT_HEIGHT
    })
    await page.goto(f"file://{html_path.resolve()}")
    await asyncio.sleep(0.5)

    total_frames = DURATION * FPS
    for i in range(total_frames):
        frame_path = TEMP_FRAME_DIR / f"frame_{i:03d}.png"
        await page.screenshot({'path': str(frame_path)})
        await asyncio.sleep(1 / FPS)

    await browser.close()


async def record_html_to_png(html_path: Path, output_png_path: Path):
    """Waits for chart animation to complete, then takes a single cropped PNG screenshot."""
    TEMP_FRAME_DIR.mkdir(parents=True, exist_ok=True)
    temp_raw_path = TEMP_FRAME_DIR / "temp_screenshot.png"

    browser = await launch(headless=True, args=["--no-sandbox"])
    page = await browser.newPage()
    await page.setViewport({
        "width": VIEWPORT_WIDTH,
        "height": VIEWPORT_HEIGHT
    })
    await page.goto(f"file://{html_path.resolve()}")

    # Wait for full animation duration plus padding
    await asyncio.sleep(DURATION + 0.5)

    await page.screenshot({'path': str(temp_raw_path)})
    await browser.close()

    # Crop and save final image
    if temp_raw_path.exists():
        img = Image.open(temp_raw_path)
        cropped = autocrop_frame(img)
        output_png_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_png_path)
        temp_raw_path.unlink()


def compile_frames_to_mp4(output_mp4_path: Path):
    output_mp4_path.parent.mkdir(parents=True, exist_ok=True)

    total_frames = DURATION * FPS
    cropped_paths = []

    for i in range(total_frames):
        raw_path = TEMP_FRAME_DIR / f"frame_{i:03d}.png"
        if raw_path.exists():
            img = Image.open(raw_path)
            cropped = autocrop_frame(img)
            crop_path = TEMP_FRAME_DIR / f"cropped_{i:03d}.png"
            cropped.save(crop_path)
            cropped_paths.append(str(crop_path))

    clip = ImageSequenceClip(cropped_paths, fps=FPS)
    clip.write_videofile(str(output_mp4_path),
                         codec="libx264",
                         audio=False,
                         bitrate="15000k",
                         preset="medium")


def cleanup_temp_frames():
    if TEMP_FRAME_DIR.exists():
        for f in TEMP_FRAME_DIR.glob("*.png"):
            f.unlink()
        TEMP_FRAME_DIR.rmdir()


def render_chart(template_name: str,
                 render_context: dict,
                 output_path: Path,
                 as_png: bool = False):
    """Renders Jinja HTML template and exports either animated MP4 or static PNG."""
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
    template = env.get_template(template_name)

    # Inject chart-theme.json directly into template context
    render_context["theme_json"] = load_theme()

    temp_html_path = CHARTS_DIR / "_temp_render.html"
    rendered_html = template.render(**render_context)

    with open(temp_html_path, "w", encoding="utf-8") as f:
        f.write(rendered_html)

    if as_png:
        print(f"[*] Rendering static PNG for {output_path.stem}...")
        asyncio.run(record_html_to_png(temp_html_path, output_path))
        print(f"[+] Static chart PNG saved to {output_path}\n")
    else:
        print(f"[*] Recording ECharts animation for {output_path.stem}...")
        asyncio.run(record_html_to_frames(temp_html_path))
        compile_frames_to_mp4(output_path)
        print(f"[+] Animated chart MP4 saved to {output_path}\n")

    cleanup_temp_frames()

    if temp_html_path.exists():
        temp_html_path.unlink()


def generate_all_channel_charts(target_handle: str,
                                output_dir_base: str = "./output_charts",
                                as_png: bool = False):
    clean_handle = target_handle.lstrip("@").lower()
    ext = "png" if as_png else "mp4"

    benchmark_path = Path(f"data/processed/{clean_handle}_benchmark.json")
    scorecard_path = Path(f"data/processed/{clean_handle}_scorecard.json")

    if not benchmark_path.exists() or not scorecard_path.exists():
        raise FileNotFoundError(
            f"Missing benchmark or scorecard for @{clean_handle} in data/processed/"
        )

    with open(benchmark_path, "r", encoding="utf-8") as f:
        bm = json.load(f)

    with open(scorecard_path, "r", encoding="utf-8") as f:
        sc = json.load(f)

    output_dir = Path(output_dir_base) / clean_handle
    output_dir.mkdir(parents=True, exist_ok=True)

    m = bm["metrics_summary"]
    p = sc["pillars"]

    # 1. DNA Radar
    render_chart("dna_radar.html", {
        "target_handle": clean_handle,
        "z_scores": {
            "toxicity": m["toxicity_rate"]["z_score"],
            "tribe": m["tribe_ratio"]["z_score"],
            "parasocial": m["parasocial_density"]["z_score"],
            "stress": m["stress_delta"]["z_score"],
            "authenticity": m["authenticity_score"]["z_score"],
        }
    },
                 output_dir / f"{clean_handle}_dna_radar.{ext}",
                 as_png=as_png)

    # 2. Peer Ranking Bar
    render_chart("peer_ranking.html", {
        "target_handle":
        clean_handle,
        "channels": [f"@{clean_handle}", "Niche Mean"],
        "values": [
            p["p2_core_tribe_ratio"]["tribe_ratio_pct"],
            m["tribe_ratio"]["niche_mean"]
        ]
    },
                 output_dir / f"{clean_handle}_peer_ranking.{ext}",
                 as_png=as_png)

    # 3. Resonance Delta Paired Bar
    render_chart(
        "resonance_delta.html", {
            "target_handle": clean_handle,
            "baseline_tox": p["p1_empathy_vs_toxicity"]["toxicity_rate_pct"],
            "top_tox": p["p1_empathy_vs_toxicity"]["top_comment_toxicity_pct"],
            "baseline_para":
            p["p3_parasocial_impact"]["parasocial_density_per_1k"],
            "top_para":
            p["p3_parasocial_impact"]["top_comment_parasocial_density"]
        },
        output_dir / f"{clean_handle}_resonance_delta.{ext}",
        as_png=as_png)

    # 4. Stress Diverging Bar
    render_chart("stress_diverging.html", {
        "target_handle":
        clean_handle,
        "channels": [f"@{clean_handle}", "Niche Mean"],
        "deltas":
        [m["stress_delta"]["target_value"], m["stress_delta"]["niche_mean"]]
    },
                 output_dir / f"{clean_handle}_stress_test.{ext}",
                 as_png=as_png)

    # 5. Peer Scatter Matrix
    candidates = sc.get("discovery", {}).get("candidates", [])
    scatter_data = [{
        "handle": f"@{c['handle']}",
        "semantic": c.get("semantic_similarity", 0.0),
        "affinity": c.get("audience_affinity", 0.0),
        "source": c.get("discovery_source", "Approach A")
    } for c in candidates]
    render_chart("peer_scatter.html", {
        "target_handle": clean_handle,
        "scatter_data": scatter_data
    },
                 output_dir / f"{clean_handle}_peer_scatter.{ext}",
                 as_png=as_png)

    # 6. Emotion Spectrum Donut
    render_chart("emotion_donut.html", {"target_handle": clean_handle},
                 output_dir / f"{clean_handle}_emotion_donut.{ext}",
                 as_png=as_png)

    # 7. Bot & Anomaly Breakdown
    auth_score = p["p5_bot_and_authenticity"]["comment_authenticity_score"]
    suspicious_pct = p["p5_bot_and_authenticity"]["suspicious_rate_pct"]
    render_chart("bot_breakdown.html", {
        "target_handle": clean_handle,
        "authenticity_score": auth_score,
        "duplicate_pct": round(suspicious_pct * 0.6, 2),
        "cluster_pct": round(suspicious_pct * 0.4, 2)
    },
                 output_dir / f"{clean_handle}_bot_breakdown.{ext}",
                 as_png=as_png)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Generate animated MP4 or static PNG ECharts visualizations for Tube-mancy analysis."
    )
    parser.add_argument("handle",
                        type=str,
                        help="Target channel handle (e.g. donflurgundy)")
    parser.add_argument("-o",
                        "--output-dir",
                        type=str,
                        default="./output_charts",
                        help="Output folder")
    parser.add_argument("--png",
                        action="store_true",
                        help="Export static PNG images instead of MP4 videos")
    args = parser.parse_args()

    generate_all_channel_charts(args.handle,
                                output_dir_base=args.output_dir,
                                as_png=args.png)
