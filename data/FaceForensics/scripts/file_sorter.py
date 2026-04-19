#Code to aid in the sorting of the videos in the corresponding train/val/test folders
#I decided on a 80/10/10 split

from pathlib import Path
import shutil
import random


SEED = 42

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

COPY_INSTEAD_OF_MOVE = True 

BASE = Path("/pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/data/FaceForensics/c23")

REAL_DIR = BASE / "original_sequences/youtube/c23/videos"
FAKE_DIR = BASE / "manipulated_sequences/Deepfakes/c23/videos"

OUT_BASE = BASE.parent / "c23_sorted"  


def make_dirs():
    for split in ["train", "val", "test"]:
        (OUT_BASE / split / "real_videos").mkdir(parents=True, exist_ok=True)
        (OUT_BASE / split / "fake_videos").mkdir(parents=True, exist_ok=True)


def split_ids(real_files):
    ids = sorted([p.stem for p in real_files])  # "183"
    random.seed(SEED)
    random.shuffle(ids)

    n = len(ids)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])

    return train_ids, val_ids, test_ids


def get_split(video_id, train_ids, val_ids, test_ids):
    if video_id in train_ids:
        return "train"
    elif video_id in val_ids:
        return "val"
    else:
        return "test"


def transfer(src, dst):
    if COPY_INSTEAD_OF_MOVE:
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)


def main():
    print("Creating folders...")
    make_dirs()

    real_files = sorted(REAL_DIR.glob("*.mp4"))
    fake_files = sorted(FAKE_DIR.glob("*.mp4"))

    print(f"Found {len(real_files)} real videos")
    print(f"Found {len(fake_files)} fake videos")

    train_ids, val_ids, test_ids = split_ids(real_files)

    print(f"Train IDs: {len(train_ids)}")
    print(f"Val IDs:   {len(val_ids)}")
    print(f"Test IDs:  {len(test_ids)}")

    # REAL
    for real_path in real_files:
        vid = real_path.stem
        split = get_split(vid, train_ids, val_ids, test_ids)
        dst = OUT_BASE / split / "real_videos" / real_path.name
        transfer(real_path, dst)

    # FAKE
    for fake_path in fake_files:
        source_id = fake_path.stem.split("_")[0]
        split = get_split(source_id, train_ids, val_ids, test_ids)
        dst = OUT_BASE / split / "fake_videos" / fake_path.name
        transfer(fake_path, dst)

    print("Video sorting sucessful")


if __name__ == "__main__":
    main()
