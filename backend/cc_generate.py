import os
import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip
import pysrt
import srt
from datetime import timedelta

def extract_audio(video_path):
    """Extracts audio from the given video file and saves it as audio.wav."""
    video = VideoFileClip(video_path)
    audio_path = "audio.wav"
    video.audio.write_audiofile(audio_path)
    print("Audio extracted and saved as", audio_path)
    return audio_path


def transcribe_audio(audio_path):
    """Transcribes the audio and returns subtitles with timestamps"""
    print("Transcribing audio...")
    model = whisper.load_model("small")  # You can change "small" to "base", "medium", etc.
    result = model.transcribe(audio_path)
    subtitles = []
    
    for segment in result["segments"]:
        start_time = timedelta(seconds=segment["start"])
        end_time = timedelta(seconds=segment["end"])
        text = segment["text"]
        subtitles.append((start_time, end_time, text))
    
    print("Transcription completed.")
    return subtitles


def create_srt(subtitles):
    """Creates an SRT subtitle file from the transcribed subtitles."""
    srt_subtitles = []
    
    for index, (start_time, end_time, text) in enumerate(subtitles, start=1):
        srt_subtitles.append(srt.Subtitle(index, start_time, end_time, text))
    
    subtitle_path = "subtitles.srt"
    with open(subtitle_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(srt_subtitles))
    
    print("Subtitles saved as", subtitle_path)
    return subtitle_path


def add_subtitles_to_video(video_path, subtitle_path, output_path="video_with_subtitles.mp4"):
    """Adds the subtitle file to the video and saves the new video"""
    output_path = os.path.join(os.path.dirname(video_path), output_path)
    command = f'ffmpeg -i "{video_path}" -vf "subtitles={subtitle_path}" -c:a copy "{output_path}"'
    
    print("Adding subtitles to the video...")
    os.system(command)
    print("Video with subtitles saved as", output_path)

def main():
    """Main function to handle video subtitle processing"""
    video_path = input("Enter the video file path: ").strip()

    if not os.path.exists(video_path):
        print("Error: Video file not found!")
        return

    audio_path = extract_audio(video_path)
    subtitles = transcribe_audio(audio_path)
    subtitle_file = create_srt(subtitles)
    add_subtitles_to_video(video_path, subtitle_file)

if __name__ == "_main_":
    main()