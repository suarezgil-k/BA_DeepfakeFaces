import cv2
from pathlib import Path


def extract_frames(video_path: str, output_dir: str, every_n_frames: int = 30, prefix: str = ""):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
       print(f"WARNING: Could not open video: {video_path}")
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
#Counts how many frames have been created
                saved_idx += 1

        frame_idx += 1

    cap.release()
#Prints out 1.How many frames were created and 2. from wich video :)
    print(f"Saved {saved_idx} frames from {video_path.name} to {output_dir}")
    return saved_idx


if __name__ == "__main__":
#Code to extract frames from specific MP4s - - - - - - - -
    #fake_video = "/pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/data/FaceForensics/c40/manipulated_sequences/Deepfakes/c40/videos/183_253.mp4"
    #fake_out = "/pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/data/FaceForensics/c40/Test_FaceFrames/fake"

    #real_video = "/pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/data/FaceForensics/c40/original_sequences/youtube/c40/videos/.mp4"   
    #real_out = "/pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/data/FaceForensics/c40_frames/real"

#extract_frames(real_video,   #fake_video for fake 
                   #real_out, #fake_out for fake
                   #every_n_frames=30, # var that determined the frame interval at which a frame is extracted 
                   #prefix="real_")  #"fake_" for fake
#- - - - - - - -- 

#Code to extract frames from all mp4s - - - - -

    #real_video_dir = Path(
    #    "/pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/data/FaceForensics/c40/original_sequences/youtube/c40/videos"
    #)

    #real_out = Path(
    #    "/pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/data/FaceForensics/c40_frames/real"
    #)

    fake_video_dir = Path(
        "/pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/data/FaceForensics/c40/manipulated_sequences/Deepfakes/c40/videos"
    )

    fake_out = Path(
        "/pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/data/FaceForensics/c40_frames/fake"
    )



    video_paths = sorted(fake_video_dir.glob("*.mp4")) #real_video_dir. for real

    #print(f"Found {len(video_paths)} videos in {real_video_dir}")

    total_saved = 0

    for video_path in video_paths:
        total_saved += extract_frames(
            video_path,
            fake_out, #real_out for real
            every_n_frames=30,
            prefix="fake_" #real_ for real
        )

    print(f"Done. Saved {total_saved} frames in total.")
