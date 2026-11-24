"""Collect and score a study described by a JSON config.

    uv run python src/sociomancy/study.py config/studies/gta6.json collect
    uv run python src/sociomancy/study.py config/studies/gta6.json collect --source trends
    uv run python src/sociomancy/study.py config/studies/gta6.json score
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from sociomancy.sources.google_trends import collect_trends
from sociomancy.sources.metacritic import collect_metacritic
from sociomancy.sources.reddit import collect_reddit
from sociomancy.sources.score import score_collected
from sociomancy.sources.youtube import collect_youtube
from sociomancy.studies.config import Study, load_study

SOURCES = ("trends", "metacritic", "reddit", "youtube")


def _with_overrides(study: Study, args: argparse.Namespace) -> Study:
    return replace(
        study,
        max_reviews=args.max_reviews if args.max_reviews is not None else study.max_reviews,
        posts_per_sub=args.posts_per_sub if args.posts_per_sub is not None else study.posts_per_sub,
        comments_per_post=(
            args.comments_per_post if args.comments_per_post is not None else study.comments_per_post
        ),
        videos_per_window=(
            args.videos_per_window if args.videos_per_window is not None else study.videos_per_window
        ),
        comments_per_video=(
            args.comments_per_video if args.comments_per_video is not None else study.comments_per_video
        ),
    )


def _run_source(name: str, study: Study, refresh: bool) -> None:
    if name == "trends":
        collect_trends(study, refresh=refresh)
    elif name == "metacritic":
        collect_metacritic(study, refresh=refresh)
    elif name == "reddit":
        collect_reddit(study, refresh=refresh)
    elif name == "youtube":
        collect_youtube(study, refresh=refresh)
    else:
        raise ValueError(name)


def collect_command(args: argparse.Namespace) -> None:
    study = _with_overrides(load_study(args.config), args)
    names = SOURCES if args.source == "all" else (args.source,)
    print(f"Study {study.title} ({study.id}) from {study.path}")
    for name in names:
        try:
            _run_source(name, study, args.refresh)
        except Exception as exc:
            print(f"  [warn] {name} failed ({exc}). Other sources still run.")


def score_command(args: argparse.Namespace) -> None:
    study = load_study(args.config)
    print(f"Study {study.title} ({study.id}) from {study.path}")
    score_collected(study, refresh=args.refresh, complaints=not args.skip_complaints)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect and score a study from a JSON config.")
    parser.add_argument("config", help="Path to a study JSON file, e.g. config/studies/gta6.json")
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="Fetch one source, or all of them")
    collect.add_argument("--source", default="all", choices=["all", *SOURCES])
    collect.add_argument("--refresh", action="store_true", help="Ignore the study's raw cache and re-query")
    collect.add_argument("--max-reviews", type=int, default=None, help="Override metacritic.max_reviews")
    collect.add_argument("--posts-per-sub", type=int, default=None, help="Override reddit.posts_per_sub")
    collect.add_argument("--comments-per-post", type=int, default=None, help="Override reddit.comments_per_post")
    collect.add_argument("--videos-per-window", type=int, default=None, help="Override youtube.videos_per_window")
    collect.add_argument("--comments-per-video", type=int, default=None, help="Override youtube.comments_per_video")
    collect.set_defaults(func=collect_command)

    score = sub.add_parser("score", help="Run toxicity, sentiment, and complaint labels")
    score.add_argument("--refresh", action="store_true", help="Re-score rows already in scored_texts.csv")
    score.add_argument("--skip-complaints", action="store_true", help="Skip bart-large-mnli")
    score.set_defaults(func=score_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
