import os
import json
import asyncio
import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
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


async def _has_frame_driver(page) -> bool:
    return await page.evaluate("() => typeof window.renderWordFrame === 'function'")


async def _seek_frame(page, progress: float) -> None:
    await page.evaluate("(t) => window.renderWordFrame(t)", progress)


async def record_html_to_frames(html_path: Path, duration: float = DURATION):
    if TEMP_FRAME_DIR.exists():
        for f in TEMP_FRAME_DIR.glob("*.png"):
            f.unlink()
    TEMP_FRAME_DIR.mkdir(parents=True, exist_ok=True)

    browser = await launch(headless=True, args=["--no-sandbox"])
    page = await browser.newPage()
    await page.setViewport({"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT})
    await page.goto(f"file://{html_path.resolve()}")
    await asyncio.sleep(0.5)

    total_frames = int(duration * FPS)
    driven = await _has_frame_driver(page)
    for i in range(total_frames):
        if driven:
            await _seek_frame(page, i / (total_frames - 1))
        frame_path = TEMP_FRAME_DIR / f"frame_{i:03d}.png"
        await page.screenshot({"path": str(frame_path)})
        if not driven:
            await asyncio.sleep(1 / FPS)

    await browser.close()


async def record_html_to_png(html_path: Path, output_png_path: Path, duration: float = DURATION):
    """Waits for chart animation to complete, then takes a single cropped PNG screenshot."""
    TEMP_FRAME_DIR.mkdir(parents=True, exist_ok=True)
    temp_raw_path = TEMP_FRAME_DIR / "temp_screenshot.png"

    browser = await launch(headless=True, args=["--no-sandbox"])
    page = await browser.newPage()
    await page.setViewport({"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT})
    await page.goto(f"file://{html_path.resolve()}")

    if await _has_frame_driver(page):
        await _seek_frame(page, 1)
    else:
        await asyncio.sleep(duration + 0.5)

    await page.screenshot({"path": str(temp_raw_path)})
    await browser.close()

    if temp_raw_path.exists():
        img = Image.open(temp_raw_path)
        cropped = autocrop_frame(img)
        output_png_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_png_path)
        temp_raw_path.unlink()


def compile_frames_to_mp4(output_mp4_path: Path, duration: float = DURATION):
    output_mp4_path.parent.mkdir(parents=True, exist_ok=True)

    total_frames = int(duration * FPS)
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
    clip.write_videofile(
        str(output_mp4_path),
        codec="libx264",
        audio=False,
        bitrate="15000k",
        preset="medium",
    )


def cleanup_temp_frames():
    if TEMP_FRAME_DIR.exists():
        for f in TEMP_FRAME_DIR.glob("*.png"):
            f.unlink()
        TEMP_FRAME_DIR.rmdir()


def render_chart(
    template_name: str,
    render_context: dict,
    output_path: Path,
    as_png: bool = False,
    duration: float = DURATION,
):
    """Renders Jinja HTML template and exports either animated MP4 or static PNG."""
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
    template = env.get_template(template_name)

    render_context["theme_json"] = load_theme()

    temp_html_path = CHARTS_DIR / "_temp_render.html"
    rendered_html = template.render(**render_context)

    with open(temp_html_path, "w", encoding="utf-8") as f:
        f.write(rendered_html)

    if as_png:
        print(f"[*] Rendering static PNG for {output_path.stem}...")
        asyncio.run(record_html_to_png(temp_html_path, output_path, duration))
        print(f"[+] Static chart PNG saved to {output_path}\n")
    else:
        print(f"[*] Recording ECharts animation for {output_path.stem}...")
        asyncio.run(record_html_to_frames(temp_html_path, duration))
        compile_frames_to_mp4(output_path, duration)
        print(f"[+] Animated chart MP4 saved to {output_path}\n")

    cleanup_temp_frames()

    if temp_html_path.exists():
        temp_html_path.unlink()


# ---------------------------------------------------------------------------
# Mode A: Channel-Specific Benchmark Charts (Original Pipeline)
# ---------------------------------------------------------------------------
def generate_all_channel_charts(
    target_handle: str,
    output_dir_base: str = "./output_charts",
    as_png: bool = False,
):
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

    render_chart(
        "dna_radar.html",
        {
            "target_handle": clean_handle,
            "z_scores": {
                "toxicity": m["toxicity_rate"]["z_score"],
                "tribe": m["tribe_ratio"]["z_score"],
                "parasocial": m["parasocial_density"]["z_score"],
                "stress": m["stress_delta"]["z_score"],
                "authenticity": m["authenticity_score"]["z_score"],
            },
        },
        output_dir / f"{clean_handle}_dna_radar.{ext}",
        as_png=as_png,
    )

    render_chart(
        "peer_ranking.html",
        {
            "target_handle": clean_handle,
            "channels": [f"@{clean_handle}", "Niche Mean"],
            "values": [
                p["p2_core_tribe_ratio"]["tribe_ratio_pct"],
                m["tribe_ratio"]["niche_mean"],
            ],
        },
        output_dir / f"{clean_handle}_peer_ranking.{ext}",
        as_png=as_png,
    )

    render_chart(
        "resonance_delta.html",
        {
            "target_handle": clean_handle,
            "baseline_tox": p["p1_empathy_vs_toxicity"]["toxicity_rate_pct"],
            "top_tox": p["p1_empathy_vs_toxicity"]["top_comment_toxicity_pct"],
            "baseline_para": p["p3_parasocial_impact"]["parasocial_density_per_1k"],
            "top_para": p["p3_parasocial_impact"]["top_comment_parasocial_density"],
        },
        output_dir / f"{clean_handle}_resonance_delta.{ext}",
        as_png=as_png,
    )

    render_chart(
        "stress_diverging.html",
        {
            "target_handle": clean_handle,
            "channels": [f"@{clean_handle}", "Niche Mean"],
            "deltas": [
                m["stress_delta"]["target_value"],
                m["stress_delta"]["niche_mean"],
            ],
        },
        output_dir / f"{clean_handle}_stress_test.{ext}",
        as_png=as_png,
    )

    candidates = sc.get("discovery", {}).get("candidates", [])
    scatter_data = [
        {
            "handle": f"@{c['handle']}",
            "semantic": c.get("semantic_similarity", 0.0),
            "affinity": c.get("audience_affinity", 0.0),
            "source": c.get("discovery_source", "Approach A"),
        }
        for c in candidates
    ]
    render_chart(
        "peer_scatter.html",
        {"target_handle": clean_handle, "scatter_data": scatter_data},
        output_dir / f"{clean_handle}_peer_scatter.{ext}",
        as_png=as_png,
    )

    render_chart(
        "emotion_donut.html",
        {"target_handle": clean_handle},
        output_dir / f"{clean_handle}_emotion_donut.{ext}",
        as_png=as_png,
    )

    auth_score = p["p5_bot_and_authenticity"]["comment_authenticity_score"]
    suspicious_pct = p["p5_bot_and_authenticity"]["suspicious_rate_pct"]
    render_chart(
        "bot_breakdown.html",
        {
            "target_handle": clean_handle,
            "authenticity_score": auth_score,
            "duplicate_pct": round(suspicious_pct * 0.6, 2),
            "cluster_pct": round(suspicious_pct * 0.4, 2),
        },
        output_dir / f"{clean_handle}_bot_breakdown.{ext}",
        as_png=as_png,
    )


# ---------------------------------------------------------------------------
# Mode B: 4-Act Topic & Trend Video Essay Charts (echarts_payload.json)
# ---------------------------------------------------------------------------
def generate_topic_charts(
    payload_path: str = "data/processed/topics/echarts_payload.json",
    output_dir_base: str = "./output_charts/topics",
    as_png: bool = False,
):
    ext = "png" if as_png else "mp4"
    p_file = Path(payload_path)
    if not p_file.exists():
        raise FileNotFoundError(
            f"Missing {p_file}. Run youtube_algo_analyzer.py first."
        )

    payload = json.loads(p_file.read_text(encoding="utf-8"))
    out_dir = Path(output_dir_base)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_act1 = pd.DataFrame(payload.get("act1_packaging_formula", []))
    df_fmt_tox = pd.DataFrame(payload.get("act2_format_toxicity_summary", []))
    df_comments = pd.DataFrame(payload.get("act2_top_comments_sample", []))
    df_ai_vol = pd.DataFrame(payload.get("act3_ai_youtube_volume", []))
    df_ai_trends = pd.DataFrame(payload.get("act3_ai_search_trends", []))
    df_decay = pd.DataFrame(payload.get("act4_decay_curves", []))

    # 1. Act 1: Runtime (Min) vs. Views Scatter by Channel Tier
    if not df_act1.empty:
        # Cap at 60 mins so extreme 4-hour livestreams don't compress the 8-14m sweet spot
        scatter_df = df_act1[df_act1["Duration_Min"] <= 60].copy()
        tiers = ["Large (500K+)", "Mid (50K-500K)", "Small (<50K)"]
        tier_series = {}
        for t in tiers:
            sub = scatter_df[scatter_df["Channel_Tier"] == t]
            tier_series[t] = [
                [
                    round(float(r.Duration_Min), 2),
                    int(r.View_Count),
                    str(r.Title)[:55],
                    str(r.Channel_Title),
                ]
                for r in sub.itertuples()
            ]

        render_chart(
            "topic_act1_runtime_scatter.html",
            {
                "video_count": int(len(df_act1)),
                "tier_series": tier_series,
            },
            out_dir / f"act1_runtime_scatter.{ext}",
            as_png=as_png,
        )

        # 2. Act 1: Title Syntax Performance (Median Views & View-to-Sub Breakout Ratio)
        hook_defs = [
            ("Breaking Hook", "Has_Breaking_Hook"),
            ("Controversy Word", "Has_Controversy_Word"),
            ("Contains Number", "Has_Number"),
            ("Question (?)", "Has_Question"),
        ]
        hook_labels = []
        median_views = []
        median_breakout = []

        for label, col in hook_defs:
            sub = df_act1[df_act1[col].astype(bool)]
            hook_labels.append(f"{label}\n(n={len(sub)})")
            median_views.append(int(sub["View_Count"].median()) if not sub.empty else 0)
            valid_ratio = sub["View_to_Sub_Ratio"].replace(0, np.nan).dropna()
            median_breakout.append(round(float(valid_ratio.median()), 2) if not valid_ratio.empty else 0.0)

        # Baseline (videos without any of the 4 hooks)
        no_hook_mask = ~(
            df_act1["Has_Breaking_Hook"].astype(bool)
            | df_act1["Has_Controversy_Word"].astype(bool)
            | df_act1["Has_Number"].astype(bool)
            | df_act1["Has_Question"].astype(bool)
        )
        base_sub = df_act1[no_hook_mask]
        hook_labels.append(f"Neutral / Plain\n(n={len(base_sub)})")
        median_views.append(int(base_sub["View_Count"].median()) if not base_sub.empty else 0)
        base_ratio = base_sub["View_to_Sub_Ratio"].replace(0, np.nan).dropna()
        median_breakout.append(round(float(base_ratio.median()), 2) if not base_ratio.empty else 0.0)

        render_chart(
            "topic_act1_title_hooks.html",
            {
                "hook_labels": hook_labels,
                "median_views": median_views,
                "median_breakout": median_breakout,
            },
            out_dir / f"act1_title_hooks.{ext}",
            as_png=as_png,
        )

    # 3. Act 2: Format Comment Rate & Resonance Toxicity Delta (ΔR_tox)
    if not df_act1.empty and "Video_Format" in df_act1.columns:
        formats = ["Opinion or Rant", "News Recap", "Developer Interview"]
        avg_comment_rates = []
        baseline_tox_pct = []
        top3_tox_pct = []
        delta_tox = []

        tox_lookup = (
            df_fmt_tox.set_index("Video_Format").to_dict(orient="index")
            if not df_fmt_tox.empty
            else {}
        )

        for fmt in formats:
            f_sub = df_act1[df_act1["Video_Format"] == fmt]
            avg_comment_rates.append(
                round(float(f_sub["Comment_Rate_Pct"].mean()), 3) if not f_sub.empty else 0.0
            )
            t_info = tox_lookup.get(fmt, {})
            baseline_tox_pct.append(round(float(t_info.get("Baseline_Avg_Toxicity", 0.0)) * 100, 1))
            top3_tox_pct.append(round(float(t_info.get("Top3_Avg_Toxicity", 0.0)) * 100, 1))
            delta_tox.append(round(float(t_info.get("Resonance_Delta_Toxicity", 0.0)) * 100, 2))

        render_chart(
            "topic_act2_format_engagement.html",
            {
                "formats": formats,
                "comment_rates": avg_comment_rates,
                "baseline_tox": baseline_tox_pct,
                "top3_tox": top3_tox_pct,
                "delta_tox": delta_tox,
            },
            out_dir / f"act2_format_engagement.{ext}",
            as_png=as_png,
        )

    # 4. Act 2: Top Comments Category Donut (Supportive vs. Critical vs. Industry/Dev)
    if not df_comments.empty and "Comment_Category" in df_comments.columns:
        counts = df_comments["Comment_Category"].value_counts()
        pie_items = [
            {"name": str(cat), "value": int(cnt)} for cat, cnt in counts.items()
        ]
        dominant_cat = pie_items[0]["name"] if pie_items else "N/A"
        render_chart(
            "topic_act2_comment_categories.html",
            {
                "total_comments": int(len(df_comments)),
                "dominant_category": dominant_cat,
                "pie_items": pie_items,
            },
            out_dir / f"act2_comment_categories.{ext}",
            as_png=as_png,
        )

    # 5. Act 3: 12-Month AI in Games Volume, Negative Title %, & Google Trends
    if not df_ai_vol.empty:
        df_ai_vol = df_ai_vol.sort_values("Month").reset_index(drop=True)
        months = df_ai_vol["Month"].astype(str).tolist()
        upload_est = [int(x) for x in df_ai_vol["Upload_Volume_Estimate"]]
        neg_pct = [round(float(x), 1) for x in df_ai_vol["Negative_Title_Pct"]]

        # Align weekly trendspy interest to YYYY-MM buckets if available
        yt_trend_monthly = [0.0] * len(months)
        if not df_ai_trends.empty and "Date" in df_ai_trends.columns:
            df_ai_trends["Month"] = df_ai_trends["Date"].astype(str).str.slice(0, 7)
            trend_col = (
                "AI in games (YouTube)"
                if "AI in games (YouTube)" in df_ai_trends.columns
                else "AI in games (Web)"
            )
            if trend_col in df_ai_trends.columns:
                m_map = df_ai_trends.groupby("Month")[trend_col].mean().to_dict()
                yt_trend_monthly = [round(float(m_map.get(m, 0.0)), 1) for m in months]

        render_chart(
            "topic_act3_ai_timeline.html",
            {
                "months": months,
                "upload_est": upload_est,
                "neg_pct": neg_pct,
                "yt_trend": yt_trend_monthly,
            },
            out_dir / f"act3_ai_timeline.{ext}",
            as_png=as_png,
        )

    # 6. Act 4: Controversy Decay Curves (Normalized % of Peak Upload Volume by Day)
    if not df_decay.empty:
        max_day = int(df_decay["Day_Offset"].max())
        day_labels = [f"Day {d}" for d in range(max_day + 1)]
        event_series = []

        for event_name, grp in df_decay.groupby("Event"):
            grp = grp.sort_values("Day_Offset")
            peak_vol = max(float(grp["Upload_Volume_Estimate"].max()), 1.0)
            day_map = grp.set_index("Day_Offset").to_dict(orient="index")

            points = []
            for d in range(max_day + 1):
                if d in day_map:
                    raw_vol = int(day_map[d]["Upload_Volume_Estimate"])
                    pct_of_peak = round((raw_vol / peak_vol) * 100, 1)
                    gdelt_cnt = int(day_map[d].get("GDELT_News_Articles", 0))
                    points.append({"value": pct_of_peak, "raw_uploads": raw_vol, "gdelt": gdelt_cnt})
                else:
                    points.append(None)

            topic_type = str(grp["Topic_Type"].iloc[0])
            event_series.append(
                {
                    "name": f"{event_name} ({topic_type})",
                    "is_regulation": "Regulation" in topic_type,
                    "data": points,
                }
            )

        render_chart(
            "topic_act4_decay_curves.html",
            {
                "day_labels": day_labels,
                "event_series": event_series,
            },
            out_dir / f"act4_decay_curves.{ext}",
            as_png=as_png,
        )


def _num_or_none(value, digits: int = 2):
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return None
    return round(float(value), digits)


def _day_label(iso_day: str) -> str:
    return pd.Timestamp(iso_day).strftime("%b %-d")


# ---------------------------------------------------------------------------
# Mode C: Study charts (config/studies/*.json → data/processed/{id}/)
# ---------------------------------------------------------------------------
def _render_vibe(timeline: pd.DataFrame, output_path: Path, as_png: bool, title: str, subtitle: str, badge: str):
    days = [_day_label(d) for d in timeline["day"]]
    launch_rows = timeline[timeline["window"] == "launch_weekend"] if "window" in timeline.columns else timeline.iloc[0:0]
    launch_label = _day_label(str(launch_rows["day"].iloc[0])) if not launch_rows.empty else days[0]
    render_chart(
        "study_act1_vibe.html",
        {
            "chart_title": title,
            "chart_subtitle": subtitle,
            "chart_badge": badge,
            "days": days,
            "launch_label": launch_label,
            "positive": [_num_or_none(v, 2) for v in timeline["positive_pct"]],
            "neutral": [_num_or_none(v, 2) for v in timeline["neutral_pct"]],
            "negative": [_num_or_none(v, 2) for v in timeline["negative_pct"]],
            "interest": [_num_or_none(v, 1) for v in timeline["interest"]] if "interest" in timeline.columns else [None] * len(days),
            "toxicity": [_num_or_none(v, 2) for v in timeline["avg_toxicity_pct"]],
        },
        output_path,
        as_png=as_png,
    )


_TREND_COLORS = ["#5cd7c1", "#a1a1aa", "#fff59c", "#f9445a", "#5d1ccd", "#3DDC84"]


def _render_trend_charts(src: Path, out_dir: Path, ext: str, as_png: bool) -> None:
    trends_path = src / "google_trends.csv"
    if trends_path.exists():
        trends = pd.read_csv(trends_path)
        skip = {"date", "interest", "window"}
        series_cols = [col for col in trends.columns if col not in skip]
        days = [_day_label(d) for d in trends["date"]] if "date" in trends.columns else []
        series = []
        for index, col in enumerate(series_cols):
            color = _TREND_COLORS[index % len(_TREND_COLORS)]
            series.append(
                {
                    "name": col,
                    "type": "line",
                    "smooth": True,
                    "connectNulls": False,
                    "symbolSize": 8,
                    "lineStyle": {"width": 4, "color": color},
                    "itemStyle": {"color": color},
                    "data": [_num_or_none(v, 1) for v in trends[col]],
                }
            )
        if series and days:
            has_youtube = any("YouTube" in col for col in series_cols)
            render_chart(
                "study_trends_keywords.html",
                {
                    "days": days,
                    "series": series,
                    "chart_title": "Search Interest: Web and YouTube" if has_youtube else "Search Interest: Web",
                    "chart_subtitle": (
                        "GTA 6 and GTA VI on one 0–100 scale, web search and YouTube search"
                        if has_youtube
                        else "GTA 6 and GTA VI on one 0–100 web-search scale"
                    ),
                },
                out_dir / f"trends_keywords.{ext}",
                as_png=as_png,
            )

    related_path = src / "google_trends_related.csv"
    if related_path.exists() and related_path.stat().st_size > 0:
        try:
            related = pd.read_csv(related_path)
        except pd.errors.EmptyDataError:
            related = pd.DataFrame()
        rising = related[related["kind"] == "rising"].copy() if not related.empty and "kind" in related.columns else pd.DataFrame()
        if not rising.empty:
            rising["value_num"] = pd.to_numeric(rising["value_num"], errors="coerce")
            present = [v for v in rising["value_num"] if pd.notna(v)]
            breakout_value = round(max(present) * 1.15, 1) if present else 100.0

            def top_queries(window: str, limit: int = 8) -> list[str]:
                part = rising[rising["window"] == window].copy()
                part["_rank"] = part["value_num"].fillna(breakout_value)
                part = part.sort_values(["_rank", "query"], ascending=[False, True])
                return [str(q) for q in part["query"].head(limit).tolist()]

            windows = [w for w in ("pre_launch", "launch_weekend") if (rising["window"] == w).any()]
            queries: list[str] = []
            for window in windows:
                for query in top_queries(window):
                    if query not in queries:
                        queries.append(query)

            def points(window: str) -> list[dict | None]:
                part = rising[rising["window"] == window]
                lookup = {str(row.query): row for row in part.itertuples()}
                built = []
                for query in queries:
                    row = lookup.get(query)
                    if row is None:
                        built.append({"value": None, "raw": ""})
                        continue
                    raw = "" if pd.isna(row.value) else str(row.value)
                    number = row.value_num
                    if pd.isna(number):
                        built.append({"value": breakout_value, "raw": raw or "Breakout"})
                    else:
                        label = raw if raw and raw.lower() != "nan" else str(int(number) if float(number).is_integer() else number)
                        built.append({"value": round(float(number), 1), "raw": label})
                return built

            if queries:
                render_chart(
                    "study_trends_related.html",
                    {
                        "queries": queries,
                        "pre": points("pre_launch"),
                        "launch": points("launch_weekend"),
                    },
                    out_dir / f"trends_related.{ext}",
                    as_png=as_png,
                )

    region_path = src / "google_trends_region.csv"
    if region_path.exists():
        region = pd.read_csv(region_path)
        if not region.empty and {"country", "interest"}.issubset(region.columns):
            region = region.sort_values("interest", ascending=False).head(12)
            render_chart(
                "study_trends_region.html",
                {
                    "countries": [str(c) for c in region["country"]],
                    "interest": [round(float(v), 1) for v in region["interest"]],
                },
                out_dir / f"trends_region.{ext}",
                as_png=as_png,
            )


_COMPLAINT_ORDER = ["Gameplay & Design", "Pricing / Value", "Technical / Performance"]
_COMPLAINT_COLORS = {
    "Gameplay & Design": "#5cd7c1",
    "Pricing / Value": "#fff59c",
    "Technical / Performance": "#f9445a",
}
_WORD_STOP = frozenset(
    """
    a an the and or but if in on of to for with at by from as is are was were be been being
    it this that these those i you we they he she me him her us them my your our their
    not no nor yes just like really very so too also about into over after before out up down
    than then there here what when who how why which can could would should will do did does
    doing done have has had having get got getting im ive id youre dont doesnt didnt cant wont
    isnt wasnt arent its it's i'm i've you're don't doesn't didn't can't won't isn't
    all more some any most much many other such only even still back way make made making
    because thing things something anything everything someone people want wanted need
    game games gta vi gtavi grand theft auto rockstar play playing played player players
    good bad best worst better feel feels felt time well now new old first last
    one see every damn that's thats real never know bro used off hours around yet far let
    where while give say same instead video fucking gonna makes whole away already own trying
    don look looks looking going go goes went come came think thought actually pretty quite
    lot lots bit kinda sort though since without another again always really ever day man
    great nothing dead miss hard waiting life hate love slow year years
    que los las por una del con esto esta son pero como para juego
    """.split()
)
_WORD_TOKEN = re.compile(r"[a-z][a-z']{2,}")
_SOURCE_NOUNS = {
    "youtube": "top comments",
    "reddit": "top items",
    "metacritic": "top reviews",
}


def _labeled_complaints(scored: pd.DataFrame) -> pd.DataFrame:
    frame = scored.copy()
    frame["complaint_category"] = frame["complaint_category"].fillna("").astype(str)
    frame = frame[frame["complaint_category"].str.len() > 0]
    if "is_top" not in frame.columns:
        frame["is_top"] = False
    elif frame["is_top"].dtype != bool:
        frame["is_top"] = frame["is_top"].astype(str).str.strip().str.lower().isin({"1", "true", "yes"})
    return frame


def _share_points(part: pd.DataFrame, total: int) -> list[dict]:
    counts = part["complaint_category"].value_counts()
    points = []
    for category in _COMPLAINT_ORDER:
        count = int(counts.get(category, 0))
        pct = round(count / total * 100, 1) if total else 0.0
        points.append({"value": pct, "count": count, "total": total})
    return points


def _render_act3(scored: pd.DataFrame, out_dir: Path, ext: str, as_png: bool) -> None:
    labeled = _labeled_complaints(scored)
    if labeled.empty:
        return

    mix_rows = [
        ("YouTube, before Nov 19", labeled[(labeled["source"] == "youtube") & (labeled["window"] == "pre_launch")]),
        ("YouTube, launch weekend", labeled[(labeled["source"] == "youtube") & (labeled["window"] == "launch_weekend")]),
        ("Reddit", labeled[labeled["source"] == "reddit"]),
        ("Metacritic", labeled[labeled["source"] == "metacritic"]),
    ]
    mix_labels = []
    mix_series = {category: [] for category in _COMPLAINT_ORDER}
    for label, part in mix_rows:
        total = len(part)
        if total == 0:
            continue
        mix_labels.append(f"{label}  (n={total})")
        for point, category in zip(_share_points(part, total), _COMPLAINT_ORDER):
            mix_series[category].append(point)
    if mix_labels:
        render_chart(
            "study_act3_mix.html",
            {
                "labels": mix_labels,
                "gameplay": mix_series["Gameplay & Design"],
                "pricing": mix_series["Pricing / Value"],
                "technical": mix_series["Technical / Performance"],
            },
            out_dir / f"act3_complaint_mix.{ext}",
            as_png=as_png,
        )

    shift_labels = []
    shift_points = []
    source_names = {"youtube": "YouTube", "reddit": "Reddit", "metacritic": "Metacritic"}
    for source in ("youtube", "reddit", "metacritic"):
        part = labeled[labeled["source"] == source]
        top = part[part["is_top"]]
        if part.empty or top.empty:
            continue
        all_counts = part["complaint_category"].value_counts()
        top_counts = top["complaint_category"].value_counts()
        for category in _COMPLAINT_ORDER:
            all_count = int(all_counts.get(category, 0))
            top_count = int(top_counts.get(category, 0))
            all_share = all_count / len(part) * 100
            top_share = top_count / len(top) * 100
            shift_labels.append(f"{source_names[source]}  ·  {category}\n{top_count} of {len(top)}")
            shift_points.append(
                {
                    "value": round(top_share - all_share, 1),
                    "topCount": top_count,
                    "topTotal": int(len(top)),
                    "allCount": all_count,
                    "allTotal": int(len(part)),
                    "topNoun": _SOURCE_NOUNS[source],
                }
            )
    if shift_points:
        span = max(abs(point["value"]) for point in shift_points)
        limit = max(10, int(span) + 8)
        render_chart(
            "study_act3_shift.html",
            {
                "labels": shift_labels,
                "points": shift_points,
                "axis_min": -limit,
                "axis_max": limit,
            },
            out_dir / f"act3_upvote_shift.{ext}",
            as_png=as_png,
        )

    words = _complaint_words(labeled, limit=28)
    if words:
        render_chart(
            "study_act3_words.html",
            {"words": words},
            out_dir / f"act3_words.{ext}",
            as_png=as_png,
            duration=5,
        )


def _complaint_words(labeled: pd.DataFrame, limit: int = 40) -> list[dict]:
    """One count per complaint. Size is how many texts use the word, not how often it repeats."""
    docs: Counter = Counter()
    by_cat: dict[str, Counter] = defaultdict(Counter)
    for text, category in zip(labeled["text"].fillna(""), labeled["complaint_category"]):
        words = {
            token
            for token in _WORD_TOKEN.findall(str(text).lower())
            if token not in _WORD_STOP
        }
        docs.update(words)
        by_cat[category].update(words)
    ranked = [(word, count) for word, count in docs.most_common(limit) if count >= 10]
    items = []
    for word, count in ranked:
        winner = max(_COMPLAINT_ORDER, key=lambda name: by_cat[name][word])
        share = by_cat[winner][word] / count
        color = _COMPLAINT_COLORS[winner] if share >= 0.5 else "#f4f4f5"
        items.append({"name": word, "value": int(count), "textStyle": {"color": color}})
    return items


def generate_study_charts(
    study_id: str,
    processed_dir: str | None = None,
    output_dir_base: str = "./output_charts",
    as_png: bool = False,
):
    """Render the Act 1–3 charts for a scored study."""
    src = Path(processed_dir) if processed_dir else Path("data/processed") / study_id
    timeline_path = src / "act1_youtube_timeline.csv"
    tiers_path = src / "act2_channel_tiers.csv"
    complaints_path = src / "act3_complaints.csv"
    scored_path = src / "scored_texts.csv"
    missing = [p.name for p in (timeline_path, tiers_path, complaints_path, scored_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {', '.join(missing)} in {src}. Run study.py score first."
        )

    ext = "png" if as_png else "mp4"
    out_dir = Path(output_dir_base) / study_id
    out_dir.mkdir(parents=True, exist_ok=True)

    timeline = pd.read_csv(timeline_path)
    _render_vibe(
        timeline,
        out_dir / f"act1_vibe_shift.{ext}",
        as_png,
        "Act 1: Pre-Launch Week vs. Launch Weekend",
        "YouTube sentiment share, comment toxicity, and Google search interest. Day is the video publish date.",
        "ACT 1 · PUBLISH DAY",
    )
    comment_path = src / "act1_comment_timeline.csv"
    if comment_path.exists() and comment_path.stat().st_size > 0:
        comment_timeline = pd.read_csv(comment_path)
        if not comment_timeline.empty and "positive_pct" in comment_timeline.columns:
            _render_vibe(
                comment_timeline,
                out_dir / f"act1_comment_vibe.{ext}",
                as_png,
                "Act 1: When the Comments Were Written",
                "Same scores, dated by the comment. Search interest is still the web index.",
                "ACT 1 · COMMENT DAY",
            )
    _render_trend_charts(src, out_dir, ext, as_png)

    tiers = pd.read_csv(tiers_path)
    tier_labels = [
        f"{row.channel_tier}\n{int(row.videos)} videos · {int(row.comments)} comments"
        for row in tiers.itertuples()
    ]
    render_chart(
        "study_act2_tiers.html",
        {
            "tiers": tier_labels,
            "baseline": [round(float(v) * 100, 2) for v in tiers["baseline_avg_toxicity"]],
            "top": [round(float(v) * 100, 2) for v in tiers["top_avg_toxicity"]],
        },
        out_dir / f"act2_channel_tiers.{ext}",
        as_png=as_png,
    )

    scored = pd.read_csv(scored_path)
    _render_act3(scored, out_dir, ext, as_png)
    source_order = ["youtube", "reddit", "metacritic"]
    source_names = {"youtube": "YouTube", "reddit": "Reddit", "metacritic": "Metacritic"}
    means = (
        scored.groupby(["source", "window"])["toxicity_score"].mean().mul(100)
    )
    pre = []
    launch = []
    display_sources = []
    for source in source_order:
        display_sources.append(source_names[source])
        pre.append(_num_or_none(means.get((source, "pre_launch")), 2))
        launch.append(_num_or_none(means.get((source, "launch_weekend")), 2))
    render_chart(
        "study_platform_toxicity.html",
        {"sources": display_sources, "pre": pre, "launch": launch},
        out_dir / f"platform_toxicity.{ext}",
        as_png=as_png,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate animated MP4 or static PNG ECharts visualizations for Sociomancy."
    )
    parser.add_argument(
        "handle",
        nargs="?",
        type=str,
        help="Target channel handle (e.g. donflurgundy). Omit when using --topics.",
    )
    parser.add_argument(
        "--topics",
        action="store_true",
        help="Render the 4-Act topic & trend charts from data/processed/topics/echarts_payload.json",
    )
    parser.add_argument(
        "--study",
        type=str,
        help="Study id (e.g. gta6). Reads data/processed/{id}/ and writes output_charts/{id}/.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./output_charts",
        help="Output folder",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Export static PNG images instead of MP4 videos",
    )
    args = parser.parse_args()

    if args.topics and args.study:
        parser.error("Pass either --topics or --study, not both.")
    if args.topics:
        out_path = "./output_charts/topics" if args.output_dir == "./output_charts" else args.output_dir
        generate_topic_charts(output_dir_base=out_path, as_png=args.png)
    elif args.study:
        generate_study_charts(args.study, output_dir_base=args.output_dir, as_png=args.png)
    elif args.handle:
        generate_all_channel_charts(
            args.handle, output_dir_base=args.output_dir, as_png=args.png
        )
    else:
        parser.error("Provide a channel handle, or pass --topics or --study.")