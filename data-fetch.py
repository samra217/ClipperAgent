#!/usr/bin/env python3
"""
Bygger en lista med populära, långa YouTube-videor (ej musik) som har
"most replayed"-heatmap, för att träna en klippmodell.

Kräver:  pip install -U yt-dlp
Kör:     python build_dataset_list.py --target 100

Utdata:
  videos.csv           en rad per vald video (id, url, titel, kanal, visningar, längd, kategori)
  heatmaps/<id>.json   heatmap-segment [{start_time, end_time, value}, ...]
"""
import argparse
import csv
import json
import time
import urllib.parse
from collections import Counter
from pathlib import Path

import yt_dlp

# Sökfrågor per kategori. Lägg till eller ändra fritt.
QUERIES = {
    "podcast": [
        "podcast full episode",
        "podcast interview full",
        "comedy podcast episode",
        "true crime podcast",
    ],
    "interview": [
        "full interview",
        "celebrity interview",
        "hot ones",
        "long form interview",
    ],
    "talk_show": [
        "late night show full interview",
        "talk show full episode",
        "panel show full episode",
    ],
    "sketch_standup": [
        "stand up comedy full special",
        "sketch comedy compilation",
        "improv comedy full show",
    ],
    "show_reality": [
        "game show full episode",
        "challenge video 24 hours",
        "reality show full episode",
    ],
    "documentary_commentary": [
        "documentary full",
        "video essay",
        "reaction commentary video",
    ],
}

# "sp=CAM=" är YouTubes sökfilter för "sortera efter visningar".
# Fungerar det inte sorterar vi ändå själva på view_count nedan.
SORT_BY_VIEWS = "CAM="


def search(query: str, n: int) -> list[dict]:
    url = "https://www.youtube.com/results?" + urllib.parse.urlencode(
        {"search_query": query, "sp": SORT_BY_VIEWS}
    )
    opts = {
        "extract_flat": True,
        "playlistend": n,
        "quiet": True,
        "skip_download": True,
        "ignoreerrors": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False) or {}
    return [e for e in (info.get("entries") or []) if e and e.get("id")]


def details(video_id: str) -> dict | None:
    opts = {"quiet": True, "skip_download": True, "noplaylist": True, "ignoreerrors": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)

def download(video_id: str, out_dir: Path) -> dict | None:
    opts = {
        "format": "bestaudio/best",
        "paths": {"home": str(out_dir / "audio")},
        "outtmpl": "%(id)s.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",   # kbps
        }],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=100, help="antal videor totalt")
    ap.add_argument("--per-query", type=int, default=40, help="sökträffar att granska per fråga")
    ap.add_argument("--min-min", type=float, default=8, help="minsta längd i minuter")
    ap.add_argument("--max-min", type=float, default=240, help="största längd i minuter")
    ap.add_argument("--per-channel", type=int, default=3, help="max videor per kanal")
    ap.add_argument("--sleep", type=float, default=1.5, help="paus mellan anrop (s)")
    ap.add_argument("--out", default="data_insamling/", help="utdatamapp")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "heatmaps").mkdir(parents=True, exist_ok=True)

    per_cat = -(-args.target // len(QUERIES))  # ceil
    seen: set[str] = set()
    per_channel: Counter = Counter()
    rows: list[dict] = []

    for cat, queries in QUERIES.items():
        cat_rows: list[dict] = []
        for q in queries:
            if len(cat_rows) >= per_cat:
                break
            print(f"[{cat}] söker: {q}")
            entries = search(q, args.per_query)
            # Sortera själva på visningar om sökfiltret inte gav ordning.
            entries.sort(key=lambda e: e.get("view_count") or 0, reverse=True)

            for e in entries:
                if len(cat_rows) >= per_cat:
                    break
                vid = e["id"]
                if vid in seen:
                    continue
                seen.add(vid)

                dur = e.get("duration")
                if dur and not (args.min_min * 60 <= dur <= args.max_min * 60):
                    continue

                time.sleep(args.sleep)
                info = details(vid)
                if not info:
                    continue

                dur = info.get("duration") or 0
                if not (args.min_min * 60 <= dur <= args.max_min * 60):
                    continue
                if "Music" in (info.get("categories") or []):
                    continue
                channel = info.get("channel_id") or info.get("uploader_id") or "?"
                if per_channel[channel] >= args.per_channel:
                    continue
                heatmap = info.get("heatmap")
                if not heatmap:
                    continue

                per_channel[channel] += 1
                (out / "heatmaps" / f"{vid}.json").write_text(json.dumps(heatmap))
                time.sleep(args.sleep)
                download(vid, out)
                row = {
                    "id": vid,
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "title": info.get("title"),
                    "channel": info.get("channel"),
                    "channel_id": channel,
                    "views": info.get("view_count"),
                    "duration_s": dur,
                    "category": cat,
                    "language": info.get("language"),
                }
                cat_rows.append(row)
                print(f"  + {row['views']:>12,} {row['title'][:70]}")
        rows.extend(cat_rows)

    rows = rows[: args.target]
    with open(out / "videos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nKlart: {len(rows)} videor -> {out / 'videos.csv'}")


if __name__ == "__main__":
    main()