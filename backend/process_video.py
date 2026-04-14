import subprocess
import os
import shutil
import numpy as np
import pickle
import librosa
import imageio_ffmpeg
from scipy.io import wavfile
from moviepy.editor import VideoFileClip, concatenate_videoclips
import moviepy.video.fx.all as vfx
from datetime import timedelta

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cheer_detector_rf.pkl")

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

# Extract features from an audio segment
def extract_features(audio_segment, sr):
    mfcc = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_segment)
    features = np.hstack([np.mean(mfcc, axis=1), np.mean(spectral_centroid), np.mean(zero_crossing_rate)])
    return features

def merge_segments(segments, min_gap=2.0):
    """
    Merge overlapping or nearby segments into a single segment.
    """
    if not segments:
        return []

    # Sort segments by start time
    segments.sort()
    merged_segments = []
    current_start, current_end = segments[0]

    for start, end in segments[1:]:
        if start <= current_end + min_gap:  # Segments are close enough to merge
            current_end = max(current_end, end)  # Extend the current segment
        else:
            merged_segments.append((current_start, current_end))
            current_start, current_end = start, end

    # Add the last segment
    merged_segments.append((current_start, current_end))
    return merged_segments


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {remaining_minutes}m"
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"


def build_video_summary(video_duration: float, cheer_segments, cheer_threshold: float, segment_duration: float) -> str:
    total_windows = int(np.ceil(video_duration / segment_duration)) if video_duration > 0 else 0
    merged_segments = merge_segments(list(cheer_segments))
    detected_windows = len(cheer_segments)
    detected_clusters = len(merged_segments)
    summary_duration = _format_duration(video_duration)

    if detected_windows == 0:
        return (
            f"This {summary_duration} video appears to be a continuous recording with no strong crowd-energy peaks above "
            f"the current detection threshold of {cheer_threshold:.2f}. The editor did not find enough high-confidence moments "
            "to build a highlight reel, so the source video is likely calmer or more evenly paced."
        )

    window_labels = []
    for start, end in merged_segments[:3]:
        window_labels.append(f"{_format_duration(start)}-{_format_duration(end)}")

    positions = []
    for start, end in merged_segments:
        mid_point = (start + end) / 2
        ratio = mid_point / max(video_duration, 1)
        if ratio < 0.33:
            positions.append("opening")
        elif ratio < 0.66:
            positions.append("middle")
        else:
            positions.append("closing")

    position_text = ", ".join(sorted(set(positions)))
    if not position_text:
        position_text = "middle"

    highlight_word = "cluster" if detected_clusters == 1 else "clusters"
    window_word = "segment" if detected_windows == 1 else "segments"

    return (
        f"This {summary_duration} video contains {detected_windows} high-energy {window_word} grouped into {detected_clusters} {highlight_word}. "
        f"The strongest moments are concentrated in the {position_text} portion of the video. "
        f"Detected highlight windows include {', '.join(window_labels)}. "
        f"Overall, the recording looks like a long-form event or match with repeated audience reactions and several clear spike moments."
    )

def _resolve_ffmpeg_binary():
    env_binary = os.getenv("FFMPEG_BINARY")
    if env_binary and os.path.exists(env_binary):
        return env_binary

    path_binary = shutil.which("ffmpeg")
    if path_binary:
        return path_binary

    # Fallback to bundled binary shipped by imageio-ffmpeg package.
    return imageio_ffmpeg.get_ffmpeg_exe()


def apply_action_track_crop(clip, vision_engine, smoothing_buffer_cls=None):
    """
    Applies smooth action tracking crop to a clip converting it to 9:16.
    """
    if not vision_engine:
        return clip.fx(vfx.crop, x_center=clip.w/2, y_center=clip.h/2, width=int(clip.h * 9/16), height=clip.h)
        
    w, h = clip.size
    target_ratio = 9/16
    target_w = int(h * target_ratio)
    
    if smoothing_buffer_cls is None:
        # Fallback to center crop if smoothing dependency isn't available.
        return clip.fx(vfx.crop, x_center=clip.w/2, y_center=clip.h/2, width=target_w, height=clip.h)

    smooth_x = smoothing_buffer_cls(window_size=30)
    
    def process_frame(get_frame, t):
        frame = get_frame(t)
        center, conf = vision_engine.get_action_center(frame)
        
        cx = center[0] if center else w / 2
        scx = smooth_x.smooth(cx)
        
        x1 = int(scx - target_w / 2)
        if x1 < 0: x1 = 0
        if x1 + target_w > w: x1 = w - target_w
            
        return frame[0:h, x1:x1+target_w]

    return clip.fl(process_frame)


def apply_video_filter(clip, filter_type):
    if not filter_type or filter_type == "none":
        return clip
        
    if filter_type == "vibrant":
        # Boost saturation and contrast
        return clip.fx(vfx.colorx, 1.25).fx(vfx.lum_contrast, 0, 0.15)
        
    elif filter_type == "cinematic":
        # Higher contrast, slightly lower saturation, subtle teal/orange tint
        # (Simplified cinematic look via color grading)
        return clip.fx(vfx.colorx, 0.9).fx(vfx.lum_contrast, 0.05, 0.2)
        
    elif filter_type == "esports_glow":
        # High brightness and contrast to make colors pop
        return clip.fx(vfx.lum_contrast, 0.1, 0.25).fx(vfx.colorx, 1.4)
        
    elif filter_type == "bw":
        # Standard black and white Dramatic look
        return clip.fx(vfx.blackwhite)
        
    return clip

def cut_and_merge_clips(video_path, cheer_segments, output_dir, export_format="16:9", vision_engine=None, video_filter="none", smoothing_buffer_cls=None):
    video = VideoFileClip(video_path)
    cheer_clips = []

    merged_segments = merge_segments(cheer_segments)

    for start, end in merged_segments:
        mid_point = (start + end) / 2
        clip_start = max(0, mid_point - 5)
        clip_end = min(video.duration, mid_point + 5)

        if clip_end - clip_start < 10:
            if clip_start == 0:
                clip_end = min(video.duration, clip_start + 10)
            else:
                clip_start = max(0, clip_end - 10)

        clip = video.subclip(clip_start, clip_end)
        
        if export_format == "9:16":
            # REUSE the shared vision engine (Turbo fix)
            clip = apply_action_track_crop(clip, vision_engine, smoothing_buffer_cls=smoothing_buffer_cls)
            
        # Apply the chosen Pro Filter
        clip = apply_video_filter(clip, video_filter)
            
        cheer_clips.append(clip)

    if cheer_clips:
        final_video = concatenate_videoclips(cheer_clips)
        output_video_path = os.path.join(output_dir, "highlight_reel.mp4")
        final_video.write_videofile(output_video_path, codec="libx264", fps=30)
        final_video.close()
        video_duration = float(video.duration or 0)
        video.close()
        return output_video_path, video_duration
    else:
        video_duration = float(video.duration or 0)
        video.close()
        return None, video_duration

# Process video file (TURBO VERSION)
def process_video(video_path, output_dir, segment_duration=2.0, cheer_threshold=0.6, use_ai=True, export_format="16:9", video_filter="none"):
    audio_path = os.path.join(output_dir, "extracted_audio.wav")

    cv2_mod = None
    vision = None
    trans = None
    smoothing_buffer_cls = None

    # Keep advanced AI modules optional so the API can boot on minimal environments.
    if use_ai:
        try:
            import cv2 as cv2_mod
            from vision_engine import VisionEngine, SmoothingBuffer
            from transcription_engine import TranscriptionEngine

            vision = VisionEngine()
            trans = TranscriptionEngine()
            smoothing_buffer_cls = SmoothingBuffer
        except Exception as import_err:
            print(f"AI modules unavailable, continuing with audio-only mode: {import_err}")
            use_ai = False
    
    ffmpeg_binary = _resolve_ffmpeg_binary()
    try:
        subprocess.run(
            [ffmpeg_binary, "-y", "-i", video_path, "-q:a", "0", "-map", "a", audio_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

    # Turbo Pass 1: Full Transcription (Much faster than segment-by-segment)
    if use_ai and trans:
        trans.load_full_transcription(audio_path)

    y, sr = librosa.load(audio_path, sr=16000)
    total_duration = librosa.get_duration(y=y, sr=sr)
    cheer_segments = []

    # Turbo optimization: Persistent Video Capture handle
    cap = cv2_mod.VideoCapture(video_path) if use_ai and cv2_mod is not None else None

    # Process each segment
    for start_time in np.arange(0, total_duration, segment_duration):
        end_time = min(start_time + segment_duration, total_duration)
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = y[start_sample:end_sample]

        if len(segment) > 0:
            features = extract_features(segment, sr).reshape(1, -1)
            audio_prob = model.predict_proba(features)[0][1]

            final_score = audio_prob
            vision_score = 0
            text_score = 0
            
            # Smart AI boost if audio signal is promising (>0.35)
            if use_ai and audio_prob > 0.35:
                # 1. Vision Check using persistent handle (Turbo)
                if cap and cap.isOpened():
                    cap.set(cv2_mod.CAP_PROP_POS_MSEC, (start_time + end_time) / 2 * 1000)
                    ret, frame = cap.read()
                    if ret:
                        _, vision_score = vision.get_action_center(frame)
                
                # 2. Transcription Check using cached segments (Turbo Pass 2)
                text_result = trans.analyze_segment(start_time, end_time)
                text_score = text_result["score"]

                # Multimodal Weighted Fusion
                final_score = (audio_prob * 0.45) + (vision_score * 0.35) + (text_score * 0.2)

            if final_score >= cheer_threshold:
                cheer_segments.append((start_time, end_time))

    if cap: cap.release()
    if os.path.exists(audio_path): os.remove(audio_path)

    # Export with shared vision engine for tracking
    output_video_path, video_duration = cut_and_merge_clips(
        video_path,
        cheer_segments,
        output_dir,
        export_format=export_format,
        vision_engine=vision,
        video_filter=video_filter,
        smoothing_buffer_cls=smoothing_buffer_cls,
    )
    
    summary_text = build_video_summary(video_duration or total_duration, cheer_segments, cheer_threshold, segment_duration)
    summary_data = {
        "video_duration_seconds": round(video_duration or total_duration, 2),
        "detected_segments": len(cheer_segments),
        "merged_clusters": len(merge_segments(list(cheer_segments))),
        "segment_duration_seconds": segment_duration,
        "cheer_threshold": cheer_threshold,
    }
    return {
        "output_video_path": output_video_path,
        "summary_text": summary_text,
        "summary_data": summary_data,
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, default="output")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    process_video(args.video, args.output)
