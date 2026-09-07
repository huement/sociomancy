# YouTube Channel Cards

This codebase has a helper script that will generate the cards for each channel its analyzing. This script is located in the `src/cards` directory.

## Commands

### Generate transparent PNG cards from custom JSON data:

```bash
uv run python src/cards/build_cards.py path/to/my_channels.json
```

### Save PNGs to a specific output folder:

```bash
uv run python src/cards/build_cards.py path/to/my_channels.json -o ./output_cards
```

### Preview custom JSON data in the browser (dev mode):

```bash
uv run python src/cards/build_cards.py src/cards/sample.json --dev
```
