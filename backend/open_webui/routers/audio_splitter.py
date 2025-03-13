# audio_splitter.py
import os
import shutil
from pydub import AudioSegment
CHUNK_DURATION_MS=600000 # 10 分鐘
BITRATE = "128k"  # 語音適合 64kbps，可調整為 96k
def audio_splitter(file_path: str,max_size: int,output_dir: str):
    audio = AudioSegment.from_file(file_path)
    total_duration = len(audio)  # in milliseconds
    split_files = [audio]

    start = 0
    index = 1
    
    while start < total_duration:
        end = min(start + CHUNK_DURATION_MS, total_duration)  # ✅【修正】確保不會超過音檔長度
        segment = audio[start:end]
        
        # 儲存分割後的音檔
        split_file_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_part{index}.mp3")
        segment.export(split_file_path, format="mp3", bitrate=BITRATE)
        split_files.append(split_file_path)
        
        start = end
        index += 1

    return split_files
