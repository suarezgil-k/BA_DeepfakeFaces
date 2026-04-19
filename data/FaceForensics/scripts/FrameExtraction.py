from pathlib import Path
import cv2

# 30 frames because ff++ videos have similar adjacent frames
EVERY_N_FRAMES = 30

BASE_SPLIT_DIR = Path(
    "/pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/data/FaceForensics/c23_sorted"
)

OUT_BASE = Path(
    "/pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/data/FaceForensics/c23_frames"
)

SPLITS = ["train", "val", "test"]
CLASSES = {
    "real": "real_videos",
    "fake": "fake_videos",
}


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    every_n_frames: int = 30,
    prefix: str = ""
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return 0

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n_frames == 0:
            out_name = output_dir / f"{prefix}{video_path.stem}_{saved_idx:04d}.jpg"
            ok = cv2.imwrite(str(out_name), frame)
            if ok:
                saved_idx += 1

        frame_idx += 1

    cap.release()
    return saved_idx


def process_folder(split: str, class_name: str, video_subdir: str):
    input_dir = BASE_SPLIT_DIR / split / video_subdir
    output_dir = OUT_BASE / split / f"{class_name}_frames"

    video_paths = sorted(input_dir.glob("*.mp4"))

    if not video_paths:
        print(f"Warning: No videos found in: {input_dir}")
        return

    print(f"\nProcessing {split}/{class_name}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Videos found: {len(video_paths)}")

    total_saved = 0
    skipped = 0 

    for i, video_path in enumerate(video_paths, start=1):
        prefix = f"{class_name}_{video_path.stem}_"
        existing_frames = list(output_dir.glob(f"{prefix}*.jpg"))

        #Had to stopo and restart code - this makes sure ther are no duplicates
        if existing_frames:
            skipped += 1
            print(f"[{i}/{len(video_paths)}] {video_path.name} -> skipped ({len(existing_frames)} frames already exist)")
            continue

        saved = extract_frames_from_video(
            video_path=video_path,
            output_dir=output_dir,
            every_n_frames=EVERY_N_FRAMES,
            prefix=f"{class_name}_"
        )
        total_saved += saved

        print(f"[{i}/{len(video_paths)}] {video_path.name} -> saved {saved} frames")

        print(f"Done: {split}/{class_name} | total newly saved frames: {total_saved} | skipped videos: {skipped}")

def main():
    print("Starting batch frame extraction...")
    print(f"Base split dir: {BASE_SPLIT_DIR}")
    print(f"Output base:    {OUT_BASE}")
    print(f"Frame interval: every {EVERY_N_FRAMES} frames")

    for split in SPLITS:
        for class_name, video_subdir in CLASSES.items():
            process_folder(split, class_name, video_subdir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
