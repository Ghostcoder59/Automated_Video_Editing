import subprocess
import os
import wave
import math
import struct
import tempfile
from datetime import timedelta

def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {remaining_minutes}m"
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"

def extract_audio(video_path, audio_path):
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        audio_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def detect_loud_segments(audio_path, chunk_duration=1.0, threshold_ratio=1.5):
    with wave.open(audio_path, 'rb') as wf:
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        chunk_frames = int(framerate * chunk_duration)
        
        rms_values = []
        for _ in range(0, n_frames, chunk_frames):
            data = wf.readframes(chunk_frames)
            if not data:
                break
            fmt = f"<{len(data)//2}h"
            samples = struct.unpack(fmt, data)
            if samples:
                rms = math.sqrt(sum(s*s for s in samples) / len(samples))
                rms_values.append(rms)
            else:
                rms_values.append(0)
    
    if not rms_values:
        return []
        
    avg_rms = sum(rms_values) / len(rms_values)
    # Give a minimum floor incase video is completely silent
    if avg_rms < 10:
        return []

    threshold = avg_rms * threshold_ratio
    
    segments = []
    for i, rms in enumerate(rms_values):
        if rms > threshold:
            segments.append((i * chunk_duration, (i + 1) * chunk_duration))
            
    return segments

def merge_segments(segments, min_gap=2.0, duration_padding=2.0):
    if not segments:
        return []
    segments.sort()
    
    padded = [(max(0, s[0] - duration_padding), s[1] + duration_padding) for s in segments]
    
    merged = []
    current_start, current_end = padded[0]
    
    for start, end in padded[1:]:
        if start <= current_end + min_gap:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
            
    merged.append((current_start, current_end))
    return merged

def process_video(video_path, output_dir, use_ai=False, export_format="16:9", video_filter="none"):
    # FFMpeg based Lite processing (no AI heavy libraries)
    segment_duration = 1.0
    audio_path = os.path.join(output_dir, "extracted_audio.wav")
    
    try:
        extract_audio(video_path, audio_path)
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None
        
    raw_segments = detect_loud_segments(audio_path, chunk_duration=segment_duration, threshold_ratio=1.6)
    merged = merge_segments(raw_segments, min_gap=2.0, duration_padding=4.0)
    
    if os.path.exists(audio_path):
        os.remove(audio_path)
        
    if not merged:
        return None
        
    cmd_dur = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
    try:
        total_duration = float(subprocess.check_output(cmd_dur, text=True).strip())
    except:
        total_duration = 0.0
        
    vf_parts = []
    if export_format == "9:16":
        vf_parts.append("crop=ih*9/16:ih:iw/2-ih*9/32:0")
        
    if video_filter == "vibrant":
        vf_parts.append("eq=saturation=1.3:contrast=1.1")
    elif video_filter == "cinematic":
        vf_parts.append("eq=contrast=1.1:saturation=0.9")
    elif video_filter == "esports_glow":
        vf_parts.append("eq=contrast=1.15:brightness=0.05:saturation=1.4")
    elif video_filter == "bw":
        vf_parts.append("hue=s=0")
        
    vf_arg = ",".join(vf_parts) if vf_parts else None
    
    segment_files = []
    for i, (start, end) in enumerate(merged):
        if end > total_duration and total_duration > 0:
            end = total_duration
            
        clip_path = os.path.join(output_dir, f"clip_{i}.mp4")
        # Ensure we don't start before 0
        safe_start = max(0, start)
        duration = end - safe_start

        cmd_clip = ["ffmpeg", "-y", "-ss", str(safe_start), "-t", str(duration), "-i", video_path]
        if vf_arg:
            cmd_clip.extend(["-vf", vf_arg, "-c:v", "libx264"])
        else:
            cmd_clip.extend(["-c:v", "copy"]) 
            
        cmd_clip.extend(["-c:a", "copy", clip_path])
        subprocess.run(cmd_clip, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        segment_files.append(clip_path)

    output_video_path = os.path.join(output_dir, "highlight_reel.mp4")
    
    concat_txt_path = os.path.join(output_dir, "concat.txt")
    with open(concat_txt_path, "w") as f:
        for clip in segment_files:
            # properly escape paths for ffmpeg concat
            f.write(f"file '{clip.replace(os.path.sep, '/')}'\n")
            
    cmd_concat = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt_path, "-c", "copy", output_video_path]
    subprocess.run(cmd_concat, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    os.remove(concat_txt_path)
    for clip in segment_files:
        if os.path.exists(clip):
            os.remove(clip)
            
    summary_text = (
        f"This {_format_duration(total_duration)} video contains {len(raw_segments)} high-energy spikes "
        f"grouped into {len(merged)} clusters. The volume-detection engine successfully isolated the loudest moments."
    )
    
    return {
        "output_video_path": output_video_path,
        "summary_text": summary_text,
        "summary_data": {
            "video_duration_seconds": round(total_duration, 2),
            "detected_segments": len(raw_segments),
            "merged_clusters": len(merged)
        }
    }
