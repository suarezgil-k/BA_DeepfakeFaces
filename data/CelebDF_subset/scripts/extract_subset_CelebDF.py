from pathlib import Path
from collections import defaultdict
import random
import shutil

random.seed(42)

#Local Path
BASE_DIR = Path(r"C:\Users\student\Documents\Bachelorarbeit\Data\CelebDF\Celeb-DF-v2")
YOUTUBE_REAL_DIR = BASE_DIR / "YouTube-real"
CELEB_REAL_DIR = BASE_DIR / "Celeb-real"
CELEB_SYNTH_DIR = BASE_DIR / "Celeb-synthesis"

OUT_DIR = BASE_DIR / "CelebDF_subset"
OUT_REAL_DIR = OUT_DIR / "real"
OUT_FAKE_DIR = OUT_DIR / "fake"

N_YOUTUBE_REAL = 250
N_CELEB_REAL = 50
N_FAKE = 300


def ensure_dirs():
    OUT_REAL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FAKE_DIR.mkdir(parents=True, exist_ok=True)


def save_list(paths, out_txt):
    with open(out_txt, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(str(p.name) + "\n")


def copy_files(files, dst_dir):
    for f in files:
        shutil.copy2(f, dst_dir / f.name)

def unique_by_name(paths):
    seen = set()
    unique = []
    for p in paths:
        if p.name not in seen:
            unique.append(p)
            seen.add(p.name)
    return unique


def sample_youtube_real():
    files = sorted(YOUTUBE_REAL_DIR.glob("*.mp4"))
    if len(files) < N_YOUTUBE_REAL:
        raise ValueError(f"Not enough YouTube-real videos: found {len(files)}, need {N_YOUTUBE_REAL}")
    chosen = random.sample(files, N_YOUTUBE_REAL)
    return chosen


def sample_celeb_real():
    files = sorted(CELEB_REAL_DIR.glob("*.mp4"))
    groups = defaultdict(list)

    #expects filename like: id3_0008.mp4
    for f in files:
        parts = f.stem.split("_")
        identity = parts[0]
        groups[identity].append(f)

    identities = list(groups.keys())
    random.shuffle(identities)

    selected = []
    id_counts = defaultdict(int)

    #1 video per id to start
    for ident in identities:
        vids = groups[ident][:]
        random.shuffle(vids)
        selected.append(vids[0])
        id_counts[ident] += 1
        if len(selected) >= N_CELEB_REAL:
            return selected

    #fill up if amount not reached 
    remaining = []
    for ident in identities:
        vids = groups[ident][1:]
        random.shuffle(vids)
        for v in vids:
            remaining.append((ident, v))

    random.shuffle(remaining)

    for ident, v in remaining:
        if len(selected) >= N_CELEB_REAL:
            break
        selected.append(v)
        id_counts[ident] += 1

    if len(selected) < N_CELEB_REAL:
        raise ValueError(f"Could only select {len(selected)} Celeb-real videos, need {N_CELEB_REAL}")

    return selected


def sample_celeb_synthesis():
    files = sorted(CELEB_SYNTH_DIR.glob("*.mp4"))
    groups = defaultdict(list)

    # expects name like id61_id60_0009.mp4
    for f in files:
        parts = f.stem.split("_")
        if len(parts) < 3:
            continue
        target = parts[0]
        donor = parts[1]
        groups[target].append((donor, f))

    targets = list(groups.keys())
    random.shuffle(targets)

    selected = []
    target_counts = defaultdict(int)
    max_per_target = 25

    #1 video per targetface to start
    for target in targets:
        donor_files = groups[target][:]
        random.shuffle(donor_files)
        selected.append(donor_files[0][1])
        target_counts[target] += 1
        if len(selected) >= N_FAKE:
            return unique_by_name(selected)

    #fil up if amount not reached
    remaining = []
    for target in targets:
        donor_files = groups[target][1:]
        random.shuffle(donor_files)
        for donor, f in donor_files:
            remaining.append((target, donor, f))

    random.shuffle(remaining)

    for target, donor, f in remaining:
        if len(selected) >= N_FAKE:
            break
        if target_counts[target] < max_per_target:
            selected.append(f)
            target_counts[target] += 1
    
    selected = unique_by_name(selected)
    selected_names = {p.name for p in selected}

    for f in files:
        if len(selected) >= N_FAKE:
            break
        if f.name not in selected_names:
            selected.append(f)
            selected_names.add(f.name)

    if len(selected) < N_FAKE:
        raise ValueError(f"Could only select {len(selected)} fake videos, need {N_FAKE}")

    return selected


def main():
    ensure_dirs()

    youtube_real = sample_youtube_real()
    celeb_real = sample_celeb_real()
    fake = sample_celeb_synthesis()

    all_real = youtube_real + celeb_real

    print(f"YouTube-real selected: {len(youtube_real)}")
    print(f"Celeb-real selected: {len(celeb_real)}")
    print(f"Total real selected: {len(all_real)}")
    print(f"Fake selected: {len(fake)}")

    copy_files(all_real, OUT_REAL_DIR)
    copy_files(fake, OUT_FAKE_DIR)

    save_list(youtube_real, OUT_DIR / "subset_youtube_real.txt")
    save_list(celeb_real, OUT_DIR / "subset_celeb_real.txt")
    save_list(fake, OUT_DIR / "subset_fake.txt")

    print(f"\nSubset created in: {OUT_DIR}")
    print(f"Real videos copied to: {OUT_REAL_DIR}")
    print(f"Fake videos copied to: {OUT_FAKE_DIR}")


if __name__ == "__main__":
    main()