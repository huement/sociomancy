import argparse
import json
import webbrowser
from pathlib import Path
from jinja2 import Template
from html2image import Html2Image

CARDS_DIR = Path(__file__).parent

def load_template() -> Template:
    template_path = CARDS_DIR / "template.html"
    with open(template_path, "r", encoding="utf-8") as f:
        return Template(f.read())

def load_channel_data(data_path: Path) -> list[dict]:
    """Loads JSON file and normalizes output to a list of dicts."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    return data

def dev_mode(data_path: Path):
    """Renders JSON data to preview.html and opens it in the browser."""
    channels = load_channel_data(data_path)
    html_out = load_template().render(channels=channels)
    preview_path = CARDS_DIR / "preview.html"

    with open(preview_path, "w", encoding="utf-8") as f:
        f.write(html_out)

    print(f"Updated preview: {preview_path}")
    webbrowser.open(preview_path.as_uri())

def prod_mode(data_path: Path, output_dir: Path):
    """Generates transparent PNG cards for each channel in the JSON file."""
    channels = load_channel_data(data_path)
    template = load_template()

    output_dir.mkdir(parents=True, exist_ok=True)
    hti = Html2Image(
        output_path=str(output_dir),
        custom_flags=["--default-background-color=00000000"]
    )

    for channel in channels:
        rendered = template.render(channels=[channel])
        handle = channel.get("handle", "channel").replace("@", "").strip()
        filename = f"{handle}_card.png"

        # Exact width and height matching CSS container dimensions
        hti.screenshot(html_str=rendered, save_as=filename, size=(580, 220))
        print(f"Generated card: {output_dir / filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate PNG cards for YouTube channels")
    parser.add_argument("json_file",
                        nargs="?",
                        help="Path to the JSON file containing channel data")
    parser.add_argument("--dev",
                        action="store_true",
                        help="Build preview.html instead of generating PNGs")
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        help=
        "Directory to save generated PNG images (default: current directory)")

    args = parser.parse_args()

    # Fallback to sample.json if running --dev without specifying a file
    if args.dev and not args.json_file:
        input_path = CARDS_DIR / "sample.json"
    elif args.json_file:
        input_path = Path(args.json_file)
    else:
        parser.error("Please provide a path to a JSON file or use --dev")

    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")

    if args.dev:
        dev_mode(input_path)
    else:
        prod_mode(input_path, Path(args.output_dir))
