import argparse
import os
import subprocess
import sys


def _is_txt(path: str) -> bool:
    return isinstance(path, str) and path.lower().endswith(".txt")


def _read_list_file(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(line)
    return items


def split_sbs(sbs_video_path: str) -> None:
    if not os.path.isfile(sbs_video_path):
        raise FileNotFoundError(f"Input video does not exist: {sbs_video_path}")

    base, _ = os.path.splitext(sbs_video_path)
    left_out  = base + "_left.mkv"
    right_out = base + "_right.mkv"

    # Left eye = first half of width (crop=w:h:x:y)
    cmd_left = [
        "ffmpeg", "-y",
        "-i", sbs_video_path,
        "-vf", "crop=iw/2:ih:0:0",
        "-c:v", "ffv1",
        "-c:a", "copy",
        left_out,
    ]

    # Right eye = second half of width
    cmd_right = [
        "ffmpeg", "-y",
        "-i", sbs_video_path,
        "-vf", "crop=iw/2:ih:iw/2:0",
        "-c:v", "ffv1",
        "-c:a", "copy",
        right_out,
    ]

    print(f"Splitting: {sbs_video_path}")
    for label, cmd, out in [("left", cmd_left, left_out), ("right", cmd_right, right_out)]:
        print(f"  Writing {label}: {out}")
        result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"ffmpeg failed for {label} eye (exit {result.returncode})")

    print(f"Done. Wrote:\n  {left_out}\n  {right_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MDVT SBS splitter — splits a side-by-side video into _left.mkv and _right.mkv (FFV1)")
    parser.add_argument("--color_video", type=str, required=True,
                        help="Path to the SBS video, or a .txt file listing one path per line")
    args = parser.parse_args()

    if _is_txt(args.color_video):
        video_list = _read_list_file(args.color_video)
        print(f"Batch mode: {len(video_list)} entries from {args.color_video}")
        for idx, vid_path in enumerate(video_list, start=1):
            print(f"\n##### [{idx}/{len(video_list)}] {vid_path} #####")
            split_sbs(vid_path)
    else:
        split_sbs(args.color_video)
