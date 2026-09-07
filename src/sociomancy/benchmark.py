import os
import warnings

# Suppress PyTorch ONNX exporter & Hugging Face deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="torch.onnx")

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PYTHONWARNINGS"] = "ignore"

import json
import time
import argparse
from pathlib import Path
import pandas as pd

from sociomancy.main import run_pipeline


def calculate_z_score(val: float, mean: float, std: float) -> float:
    if std == 0:
        return 0.0
    return round((val - mean) / std, 2)


def ensure_control_channels_scored(target_handle: str,
                                   delay_seconds: float = 3.0) -> list[str]:
    clean_target = target_handle.lstrip("@").lower()
    target_scorecard_path = Path(
        f"data/processed/{clean_target}_scorecard.json")

    if not target_scorecard_path.exists():
        raise FileNotFoundError(
            f"Target scorecard not found at {target_scorecard_path}. Run sociomancy @{clean_target} first."
        )

    with open(target_scorecard_path, "r", encoding="utf-8") as f:
        target_data = json.load(f)

    discovery_data = target_data.get("discovery", {})
    candidates = discovery_data.get("candidates", [])

    if not candidates:
        print(
            f"[*] No discovery candidate handles found in {target_scorecard_path}. Proceeding with existing scorecards."
        )
        return []

    control_handles = []
    processed_dir = Path("data/processed")

    print(
        f"[*] Checking {len(candidates)} control group channels for cached scorecards..."
    )

    for cand in candidates:
        raw_handle = cand.get("handle") or cand.get("channel_title", "")
        cand_handle = raw_handle.lstrip("@").lower().replace(" ", "")

        if not cand_handle or cand_handle == clean_target:
            continue

        control_handles.append(cand_handle)
        scorecard_file = processed_dir / f"{cand_handle}_scorecard.json"

        if not scorecard_file.exists():
            print(
                f"\n[*] Missing scorecard for control channel @{cand_handle}. Running sociomancy pipeline..."
            )
            try:
                run_pipeline(cand_handle)
                print(
                    f"[+] Successfully generated scorecard for @{cand_handle}")
            except Exception as e:
                print(f"[!] Failed to score candidate @{cand_handle}: {e}")

            if delay_seconds > 0:
                print(
                    f"[*] Pacing requests: waiting {delay_seconds}s before next channel..."
                )
                time.sleep(delay_seconds)
        else:
            print(f" [+] Scorecard cached for @{cand_handle}")

    return control_handles


def generate_niche_benchmark(target_handle: str,
                             delay_seconds: float = 3.0) -> dict:
    clean_target = target_handle.lstrip("@").lower()
    processed_dir = Path("data/processed")

    ensure_control_channels_scored(clean_target, delay_seconds=delay_seconds)

    scorecard_files = list(processed_dir.glob("*_scorecard.json"))
    if not scorecard_files:
        raise FileNotFoundError("No scorecards found in data/processed/.")

    records = []
    target_record = None

    for sf in scorecard_files:
        with open(sf, "r", encoding="utf-8") as f:
            data = json.load(f)
            handle = data["handle"]
            p = data["pillars"]

            rec = {
                "handle":
                handle,
                "title":
                data["channel_title"],
                "subscribers":
                data.get("subscribers", 0),
                "toxicity_rate":
                p["p1_empathy_vs_toxicity"]["toxicity_rate_pct"],
                "resonance_delta_tox":
                p["p1_empathy_vs_toxicity"].get("resonance_delta_toxicity",
                                                0.0),
                "tribe_ratio":
                p["p2_core_tribe_ratio"]["tribe_ratio_pct"],
                "parasocial_density":
                p["p3_parasocial_impact"]["parasocial_density_per_1k"],
                "resonance_delta_parasocial":
                p["p3_parasocial_impact"].get("resonance_delta_parasocial",
                                              0.0),
                "stress_delta":
                p["p4_cold_audience_stress_test"]["delta_toxicity"],
                "authenticity_score":
                p.get("p5_bot_and_authenticity",
                      {}).get("comment_authenticity_score", 100.0),
            }
            records.append(rec)
            if handle == clean_target:
                target_record = rec

    if not target_record:
        raise ValueError(
            f"Target handle @{clean_target} not found in processed scorecards."
        )

    df = pd.DataFrame(records)
    total_channels = len(df)

    metrics = [
        "toxicity_rate", "resonance_delta_tox", "tribe_ratio",
        "parasocial_density", "resonance_delta_parasocial", "stress_delta",
        "authenticity_score"
    ]
    stats = {}

    for m in metrics:
        mean_val = float(df[m].mean())
        std_val = float(df[m].std()) if total_channels > 1 else 0.0
        target_val = float(target_record[m])
        z_score = calculate_z_score(target_val, mean_val, std_val)

        stats[m] = {
            "target_value": target_val,
            "niche_mean": round(mean_val, 2),
            "niche_std": round(std_val, 2),
            "z_score": z_score,
        }

    # Generate Script Narrative Takeaways
    script_hooks = []

    # Toxicity Hook
    tox_z = stats["toxicity_rate"]["z_score"]
    if abs(tox_z) >= 1.0:
        direction = "higher" if tox_z > 0 else "lower"
        script_hooks.append(
            f"Toxicity is {abs(tox_z)} standard deviations {direction} than the niche average "
            f"({stats['toxicity_rate']['target_value']}% vs niche avg of {stats['toxicity_rate']['niche_mean']}%)."
        )

    # Toxicity Resonance Hook (Upvote Amplification vs Moderation Buffer)
    res_tox_z = stats["resonance_delta_tox"]["z_score"]
    if abs(res_tox_z) >= 1.0:
        if res_tox_z > 0:
            script_hooks.append(
                f"Community upvoting actively amplifies top-comment hostility "
                f"(ΔR_tox: {stats['resonance_delta_tox']['target_value']:+} pts vs niche avg {stats['resonance_delta_tox']['niche_mean']:+} pts)."
            )
        else:
            script_hooks.append(
                f"Community upvoting acts as a buffer filtering out baseline toxicity in top comments "
                f"(ΔR_tox: {stats['resonance_delta_tox']['target_value']:+} pts vs niche avg {stats['resonance_delta_tox']['niche_mean']:+} pts)."
            )

    # Core Tribe Hook
    tribe_z = stats["tribe_ratio"]["z_score"]
    if abs(tribe_z) >= 1.0:
        direction = "stronger" if tribe_z > 0 else "weaker"
        script_hooks.append(
            f"Core viewer retention/tribe ratio is {abs(tribe_z)} std dev {direction} than peers "
            f"({stats['tribe_ratio']['target_value']}% vs niche avg of {stats['tribe_ratio']['niche_mean']}%)."
        )

    # Parasocial Resonance Hook
    res_p_z = stats["resonance_delta_parasocial"]["z_score"]
    if res_p_z >= 1.0:
        script_hooks.append(
            f"Top comment upvoting strongly amplifies parasocial connection "
            f"(ΔR_parasocial: {stats['resonance_delta_parasocial']['target_value']:+} vs niche average {stats['resonance_delta_parasocial']['niche_mean']:+})."
        )

    # Bot / Authenticity Hook
    auth_z = stats["authenticity_score"]["z_score"]
    if auth_z <= -1.0:
        script_hooks.append(
            f"Comment Authenticity is {abs(auth_z)} std dev BELOW control group average "
            f"({stats['authenticity_score']['target_value']}% vs niche average of {stats['authenticity_score']['niche_mean']}%)."
        )

    benchmark_data = {
        "target_handle": clean_target,
        "control_group_size": total_channels,
        "metrics_summary": stats,
        "script_takeaways": script_hooks
    }

    out_path = processed_dir / f"{clean_target}_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)

    return benchmark_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Auto-score missing discovered control group channels and generate Z-Score Benchmarks."
    )
    parser.add_argument("handle",
                        type=str,
                        help="Target channel handle (e.g. donflurgundy)")
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help=
        "Delay in seconds between scoring candidate channels (default: 3.0)")
    args = parser.parse_args()

    try:
        bm = generate_niche_benchmark(args.handle, delay_seconds=args.delay)
        print(f"\n==================================================")
        print(
            f"   NICHE BENCHMARK & SCRIPT HOOKS: @{bm['target_handle'].upper()}"
        )
        print(f"==================================================")
        print(
            f"Control Group Size: {bm['control_group_size']} channels sampled\n"
        )

        print("METRIC STATISTICAL MATRIX:")
        for metric, data in bm["metrics_summary"].items():
            print(
                f" • {metric.upper():<25} | Target: {data['target_value']:<6} | Mean: {data['niche_mean']:<6} | Z-Score: {data['z_score']:+}σ"
            )

        print("\n--------------------------------------------------")
        print("SCRIPT NARRATIVE TAKEAWAYS:")
        if bm["script_takeaways"]:
            for hook in bm["script_takeaways"]:
                print(f" [+] \"{hook}\"")
        else:
            print(
                " [*] Target channel sits within 1.0 std dev of niche norm across all metrics."
            )
        print("==================================================")
        print(
            f"[+] Benchmark exported to data/processed/{bm['target_handle']}_benchmark.json\n"
        )

    except Exception as e:
        print(f"[!] Benchmark Error: {e}")
