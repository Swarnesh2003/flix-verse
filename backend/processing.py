import os
from pydub import AudioSegment

# Define source and output folders
input_folder = "C:/Users/Swarneshwar S/Desktop/FILES/PROJECTS/DISH Hackathon/dataset/test"
output_folder = "C:/Users/Swarneshwar S/Desktop/FILES/PROJECTS/DISH Hackathon/dataset/test_converted"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Common audio settings
TARGET_FORMAT = "wav"
TARGET_SAMPLE_RATE = 22050
TARGET_CHANNELS = 1  # Mono

def convert_audio(file_path, output_path):
    try:
        # Load audio
        audio = AudioSegment.from_file(file_path)
        
        # Convert to mono and set sample rate
        audio = audio.set_channels(TARGET_CHANNELS).set_frame_rate(TARGET_SAMPLE_RATE)
        
        # Save in the target format
        audio.export(output_path, format=TARGET_FORMAT)
        print(f"Converted: {file_path} → {output_path}")
    except Exception as e:
        print(f"Error converting {file_path}: {e}")

# Process all audio files in the input folder
for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)

    # Skip non-audio files
    if not filename.lower().endswith((".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a")):
        continue

    # Define output path
    output_filename = os.path.splitext(filename)[0] + f".{TARGET_FORMAT}"
    output_path = os.path.join(output_folder, output_filename)

    # Convert the file
    convert_audio(input_path, output_path)

print("Conversion complete!")
