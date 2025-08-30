import pandas as pd
import os
import pandas as pd
import os
import subprocess
from tqdm import tqdm
import yt_dlp 

# Define paths
# annotation_dir = r'C:\Users\lahir\data\kinetics400\annotations'
# output_dir = r'C:\Users\lahir\data\kinetics400'
# target_classes = ["applauding"]  # Your chosen classes

# # Filter validation or test set annotations
# for split in ["val"]:
#     csv_file = os.path.expanduser(f"{annotation_dir}/{split}.csv")
#     df = pd.read_csv(csv_file)
#     filtered_df = df[df["label"].isin(target_classes)]
#     os.makedirs(output_dir, exist_ok=True)
#     filtered_df.to_csv(f"{output_dir}/{split}_filtered.csv", index=False)
#     print(f"Filtered {split} set: {len(filtered_df)} videos for {target_classes}")



def download_video(youtube_id, start_time, end_time, output_path, class_name):
    """Download and trim a single video clip using youtube-dl and ffmpeg."""
    output_file = os.path.join(output_path, class_name, f"{youtube_id}_{start_time:06d}_{end_time:06d}.mp4")
    if os.path.exists(output_file):
        print(f"Video {output_file} already exists, skipping.")
        return True
    ydl_opts = {
        "format": "bestvideo[height<=480]+bestaudio/best[height<=480]",
        "outtmpl": output_file + ".%(ext)s",
        "quiet": True,
        "merge_output_format": "mp4"
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={youtube_id}"])
        # Trim video to specified start/end times
        subprocess.run([
            "ffmpeg", "-y", "-i", f"{output_file}.mp4", "-ss", str(start_time), "-to", str(end_time),
            "-c", "copy", output_file
        ], check=True, capture_output=True)
        os.remove(f"{output_file}.mp4")  # Remove untrimmed file
        return True
    except Exception as e:
        print(f"Failed to download {youtube_id}: {e}")
        return False

# Define paths
output_dir = os.path.expanduser(r'C:\Users\lahir\data\kinetics400')
annotation_dir = os.path.expanduser(r'C:\Users\lahir\data\kinetics400')
split = "val"  # Change to "test" for test set

# Load (filtered) annotations
csv_file = f"{annotation_dir}/{split}_filtered.csv"
df = pd.read_csv(csv_file)

# Download videos
for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Downloading {split} videos"):
    youtube_id = row["youtube_id"]
    start_time = row["time_start"]
    end_time = row["time_end"]
    class_name = row["label"]
    class_dir = os.path.join(output_dir, split, class_name)
    os.makedirs(class_dir, exist_ok=True)
    download_video(youtube_id, start_time, end_time, output_dir, class_name)

