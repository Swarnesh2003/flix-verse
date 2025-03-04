import os
from datetime import datetime
import cv2
import numpy as np
import subprocess
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os
import numpy as np
import pandas as pd
import librosa
import joblib
from pydub import AudioSegment
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
warnings.filterwarnings('ignore')

import os
import time
from PIL import Image
import google.generativeai as genai

# Configure API key
genai.configure(api_key="AIzaSyD2l31nO-4ollNgy8bdtBDq8TPeoYLTa_M")

# Initialize model
model = genai.GenerativeModel("gemini-1.5-flash")

def extract_scene_frames(video_path, scene_list, output_folder, interval=2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    extracted_frames = []

    for scene_num, (start_time, end_time) in enumerate(scene_list):
        scene_folder = os.path.join(output_folder, f"scene_{scene_num}")
        os.makedirs(scene_folder, exist_ok=True)

        scene_start_sec = start_time.get_seconds()
        scene_end_sec = end_time.get_seconds()

        # Generate timestamps at `interval` seconds
        frame_times = np.arange(scene_start_sec, scene_end_sec, interval)

        for frame_time in frame_times:
            frame_idx = int(frame_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_path = os.path.join(scene_folder, f"frame_{frame_idx}.jpg")
                cv2.imwrite(frame_path, frame)
                extracted_frames.append((frame_path, frame_time))

    cap.release()
    return extracted_frames
# Define feature names for consistency
feature_names = ['mfcc_0', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 
                'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'zcr', 'rmse', 
                'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
                'spectral_contrast_0', 'spectral_contrast_1', 'spectral_contrast_2',
                'spectral_contrast_3', 'spectral_contrast_4', 'spectral_contrast_5',
                'spectral_contrast_6', 'chroma_0', 'chroma_1', 'chroma_2', 'chroma_3',
                'chroma_4', 'chroma_5', 'chroma_6', 'chroma_7', 'chroma_8', 'chroma_9',
                'chroma_10', 'chroma_11', 'mel_0', 'mel_1', 'mel_2', 'mel_3', 'mel_4',
                'mel_5', 'mel_6', 'mel_7', 'mel_8', 'mel_9', 'mel_10', 'mel_11', 'mel_12',
                'tonnetz_0', 'tonnetz_1', 'tonnetz_2', 'tonnetz_3', 'tonnetz_4', 'tonnetz_5',
                'tempo']

def resolve_close_predictions(class_probabilities, threshold, file_path):
    threshold  = 0.2
    """
    Resolve prediction when the difference between top categories is less than the threshold
    
    Args:
        class_probabilities: Dictionary with class names as keys and probabilities as values
        threshold: The threshold difference (as a decimal) to consider predictions as "close"
        
    Returns:
        The selected class name
    
    print(f"Close prediction detected (difference < {threshold*100}%). Running tiebreaker...")
    
    # Sort probabilities in descending order
    sorted_probs = sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)
    
    # Check if the difference between top two is less than threshold
    if len(sorted_probs) >= 2:
        top_class, top_prob = sorted_probs[0]
        second_class, second_prob = sorted_probs[1]
        
        if (top_prob - second_prob) < threshold:
            print(f"Top predictions too close: {top_class} ({top_prob:.2%}) vs {second_class} ({second_prob:.2%})")
            
            # Get top contenders (all classes with probability within threshold of top class)
            contenders = [cls for cls, prob in sorted_probs if (top_prob - prob) < threshold]
            
            # For this example, let's use a weighted random selection
            # In a real implementation, you might use more sophisticated methods
            contender_probs = [class_probabilities[cls] for cls in contenders]
            total = sum(contender_probs)
            normalized_probs = [p/total for p in contender_probs]
            
            # Weighted random selection
            selected_class = np.random.choice(contenders, p=normalized_probs)
            
            print(f"Tiebreaker selected: {selected_class}")
            return selected_class
    
    # If no close prediction, return the top class
    """
    # Ensure the folder exists
    print("FILE: ",file_path)
    scene_folder =  convert_path(file_path) # Change this to the desired folder
    print("Scene Folder: ", scene_folder)
    if not os.path.isdir(scene_folder):
        print(f"Error: Folder '{scene_folder}' does not exist.")
    else:
        # Get images from the folder
        image_files = get_images_from_folder(scene_folder)
        if not image_files:
            print("No images found in the folder.")
        else:
            # Read images
            images = [Image.open(img) for img in image_files]

            # Define the prompt
            prompt = "These images are different parts of a single scene. Respond with 'fights'. If there is a song, then respond with 'songs'.if there is a fight happening. If the scene is conversational respond with 'dialogues'"

            # Generate response
            response = model.generate_content([prompt] + images)
            print("Scene Name:", os.path.basename(scene_folder))
            print("Response:", response.text)
    return response.text.strip()

def convert_audio(file_path, target_format="wav", sample_rate=22050, channels=1):
    """
    Convert audio file to desired format, sample rate and channels
    """
    print(f"Converting audio: {file_path}")
    try:
        # Generate output filename
        output_path = os.path.splitext(file_path)[0] + f".{target_format}"
        
        # Load audio
        audio = AudioSegment.from_file(file_path)
        
        # Convert to mono and set sample rate
        audio = audio.set_channels(channels).set_frame_rate(sample_rate)
        
        # Save in the target format
        audio.export(output_path, format=target_format)
        print(f"✓ Conversion complete: {output_path}")
        return output_path
    except Exception as e:
        print(f"× Error converting audio: {e}")
        return None

def extract_features(file_path):
    """
    Extract audio features from a file
    """
    print(f"Extracting features from: {file_path}")
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=22050)

        # Extract features
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rmse = np.mean(librosa.feature.rms(y=y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=13), axis=1)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr), axis=1)

        # Extract Tempo (BPM)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]

        # Combine all features into a single array
        features = np.hstack([
            mfccs, zcr, rmse, spectral_centroid, spectral_bandwidth, spectral_rolloff, 
            spectral_contrast, chroma_stft, mel_spectrogram, tonnetz, tempo
        ])
        
        print(f"✓ Feature extraction complete: {len(features)} features extracted")
        return features
    
    except Exception as e:
        print(f"× Error extracting features: {e}")
        return None

def train_model(csv_path="audio_features.csv", model_path="audio_classifier.joblib"):
    """
    Train the Random Forest model on the provided CSV file
    """
    print(f"Training model using data from: {csv_path}")
    
    try:
        # Load dataset
        df = pd.read_csv(csv_path)
        
        # Separate features and labels
        X = df[feature_names]
        y = df['label']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train the model
        print("Training Random Forest Classifier...")
        rf_classifier = RandomForestClassifier(n_estimators=100, 
                                             random_state=42,
                                             n_jobs=-1)
        rf_classifier.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred = rf_classifier.predict(X_test)
        
        # Print model performance
        print("\nModel Performance:")
        print("=================")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Feature importance
        feature_imp = pd.DataFrame({'feature': feature_names, 
                                  'importance': rf_classifier.feature_importances_})
        feature_imp = feature_imp.sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_imp.head(10))
        
        # Save the model
        joblib.dump(rf_classifier, model_path)
        print(f"\nModel saved as '{model_path}'")
        
        return rf_classifier
    
    except Exception as e:
        print(f"× Error training model: {e}")
        return None
# Function to get images from a folder
def get_images_from_folder(folder_path):
    return sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

def predict_audio_class(features, model, file_path):
    """
    Predict audio class using the trained model
    """
    print(f"Making prediction using provided model")
    try:
        # Convert features to 2D array
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)
        probabilities = model.predict_proba(features_array)
        
        # Get probability scores
        class_probabilities = dict(zip(model.classes_, probabilities[0]))
        
        # Check if prediction is close (difference < 20%)
        threshold = 0.2
        sorted_probs = sorted(class_probabilities.values(), reverse=True)
        
        if len(sorted_probs) >= 2 and (sorted_probs[0] - sorted_probs[1]) < threshold:
            # Use tiebreaker when predictions are close
            final_prediction = resolve_close_predictions(class_probabilities, threshold, file_path)
        else:
            final_prediction = prediction[0]
        if(final_prediction!=prediction[0]):
            print("retrain")
            success = add_to_dataset(features, final_prediction, csv_path)
            if success:
                new_model = train_model(csv_path, model_path)
                if new_model is not None:
                    model = new_model
        print(f"✓ Prediction complete: {final_prediction}")
        print(f"✓ probabilities: {class_probabilities}")
        return final_prediction, class_probabilities
    
    except Exception as e:
        print(f"× Error making prediction: {e}")
        return None, None

def add_to_dataset(features, correct_label, dataset_path):
    """
    Add extracted features and correct label to the dataset
    """
    print(f"Adding new data point with label '{correct_label}' to dataset: {dataset_path}")
    
    try:
        # Create a dictionary with feature names and values
        data_dict = {feature_name: value for feature_name, value in zip(feature_names, features)}
        data_dict['label'] = correct_label
        
        # Create a new dataframe with this data
        new_row = pd.DataFrame([data_dict])
        
        # Check if dataset exists
        if os.path.exists(dataset_path):
            # Read existing dataset
            df = pd.read_csv(dataset_path)
            
            # Check if columns match
            if set(df.columns) != set(new_row.columns):
                print(f"× Error: Dataset columns don't match the features being added.")
                return False
                
            # Append new data
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            # Create new dataset
            df = new_row
        
        # Save updated dataset
        df.to_csv(dataset_path, index=False)
        print(f"✓ Dataset updated successfully. Total records: {len(df)}")
        return True
        
    except Exception as e:
        print(f"× Error adding to dataset: {e}")
        return False

def cleanup_temp_files(files):
    """
    Clean up temporary files
    """
    for file in files:
        try:
            if os.path.exists(file):
                os.remove(file)
                print(f"Removed temporary file: {file}")
        except Exception as e:
            print(f"Error removing file {file}: {e}")

def process_audio_file(file_path, model, csv_path=None, collect_feedback=False):
    """
    Process a single audio file and make a prediction
    """
    print(f"\nProcessing: {os.path.basename(file_path)}")
    
    temp_files = []
    
    # Step 1: Convert audio to proper format if needed
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext != '.wav':
        converted_file = convert_audio(file_path)
        if converted_file is None:
            return None
        temp_files.append(converted_file)
        audio_file_for_features = converted_file
    else:
        audio_file_for_features = file_path
    
    # Step 2: Extract features
    features = extract_features(audio_file_for_features)
    if features is None:
        cleanup_temp_files(temp_files)
        return None
    
    # Step 3: Make prediction
    prediction, probabilities = predict_audio_class(features, model, file_path)
    if prediction is None:
        cleanup_temp_files(temp_files)
        return None
    
    # Display simple results
    print(f"{os.path.basename(file_path)} - {prediction}")
    
    # If feedback collection is enabled
    if collect_feedback and csv_path:
        print("\nClass probabilities:")
        for class_name, prob in probabilities.items():
            print(f"{class_name}: {prob:.2%}")
            
        
    
    # Clean up temporary files
    cleanup_temp_files(temp_files)
    
    return {
        "file": os.path.basename(file_path),
        "prediction": prediction,
        "probabilities": probabilities
    }

def analyze_audio_folder(folder_path, csv_path="audio_features.csv", model_path="audio_classifier.joblib", collect_feedback=False):
    """
    Analyze all audio files in a folder
    """
    print(f"\n{'=' * 50}")
    print(f"AUDIO SCENE CLASSIFIER - FOLDER")
    print(f"{'=' * 50}")
    
    # Step 1: Train or load the model
    if os.path.exists(model_path):
        print(f"Loading existing model: {model_path}")
        model = joblib.load(model_path)
    else:
        print(f"Training new model from: {csv_path}")
        model = train_model(csv_path, model_path)
        if model is None:
            return None
    
    # Check if folder exists
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"× Error: The folder '{folder_path}' does not exist.")
        return None
    
    # Get all audio files in the folder
    audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac']
    audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                  if os.path.isfile(os.path.join(folder_path, f)) and
                  os.path.splitext(f)[1].lower() in audio_extensions]
    
    if not audio_files:
        print(f"× No audio files found in the folder '{folder_path}'.")
        return None
    
    print(f"\nFound {len(audio_files)} audio files. Starting analysis...")
    
    # Process each audio file
    results = []
    for audio_file in audio_files:
        result = process_audio_file(audio_file, model, csv_path if collect_feedback else None, collect_feedback)
        if result:
            results.append(result)
    
    # Display summary
    print(f"\n{'=' * 50}")
    print(f"SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total files analyzed: {len(results)}/{len(audio_files)}")
    
    # Count by class
    if results:
        class_counts = {}
        for result in results:
            prediction = result["prediction"]
            class_counts[prediction] = class_counts.get(prediction, 0) + 1
        
        print("\nPrediction counts:")
        for class_name, count in class_counts.items():
            print(f"{class_name}: {count} files ({count/len(results):.1%})")
    
    # Ask if user wants to retrain the model if feedback was collected
    if collect_feedback and csv_path:
        print("\nWould you like to retrain the model with any collected feedback? (y/n): ")
        retrain = input().strip().lower()
        
        if retrain == 'y':
            # Retrain the model
            new_model = train_model(csv_path, model_path)
    
    return results

def get_supported_audio_extensions():
    """
    Returns a list of supported audio file extensions
    """
    return ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac']

# ✅ Detect scenes
def detect_scenes(video_path, threshold=20.0, min_scene_length=3):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"❌ Video file '{video_path}' not found.")

    video = open_video(video_path)

    if not video.frame_rate:
        raise ValueError("❌ Failed to retrieve video FPS. Ensure the file is valid.")

    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    scene_manager.detect_scenes(video)
    original_scene_list = scene_manager.get_scene_list()

    if not original_scene_list:
        print("⚠ No scenes detected. Try lowering the threshold.")
        return []

    # Filter scenes (remove very short ones)
    filtered_scenes = [
        (start_time, end_time) for start_time, end_time in original_scene_list
        if (end_time.get_seconds() - start_time.get_seconds()) >= min_scene_length
    ]

    # Ensure no gaps between scenes
    continuous_scenes = []
    for i, (start, end) in enumerate(filtered_scenes):
        if i < len(filtered_scenes) - 1:
            next_start = filtered_scenes[i + 1][0]
            continuous_scenes.append((start, next_start))
        else:
            continuous_scenes.append((start, end))
    scenes_dict = {
        f"scene_{i}.mp3": (start.get_timecode(), end.get_timecode())
        for i, (start, end) in enumerate(continuous_scenes)
    }
    print("\n🎬 Detected Continuous Scenes (Start - End):")
    for i, (start, end) in enumerate(continuous_scenes):
        print(f"Scene {i}: {start.get_timecode()} → {end.get_timecode()}")

    return continuous_scenes, scenes_dict


# ✅ Extract audio for each scene using ffmpeg
def extract_audio_from_scenes(video_path, scene_list, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    extracted_audio_files = []

    for scene_num, (start_time, end_time) in enumerate(scene_list):
        start_sec = start_time.get_seconds()
        end_sec = end_time.get_seconds()
        duration = end_sec - start_sec

        if duration <= 0:
            continue  # Skip invalid scenes

        audio_path = os.path.join(output_folder, f"scene_{scene_num}.mp3")

        # Use ffmpeg to extract the audio
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", video_path,                # Input video file
            "-ss", str(start_sec),           # Start time
            "-t", str(duration),             # Duration
            "-q:a", "0",                     # Best quality
            "-map", "a",                     # Extract only audio
            audio_path                        # Output file
        ]

        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        extracted_audio_files.append(audio_path)

        print(f"✅ Extracted audio: {audio_path}")

    return extracted_audio_files

def convert_path(file_path):
    filename, _ = os.path.splitext(os.path.basename(file_path))  # Extract filename without extension
    return f"frames\{filename}"

def extract_scene_frames(video_path, scene_list, output_folder, interval=2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    extracted_frames = []

    for scene_num, (start_time, end_time) in enumerate(scene_list):
        scene_folder = os.path.join(output_folder, f"scene_{scene_num}")
        os.makedirs(scene_folder, exist_ok=True)

        scene_start_sec = start_time.get_seconds()
        scene_end_sec = end_time.get_seconds()

        # Generate timestamps at interval seconds
        frame_times = np.arange(scene_start_sec, scene_end_sec, interval)

        for frame_time in frame_times:
            frame_idx = int(frame_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_path = os.path.join(scene_folder, f"frame_{frame_idx}.jpg")
                cv2.imwrite(frame_path, frame)
                extracted_frames.append((frame_path, frame_time))

    cap.release()
    return extracted_frames

def merge_scenes(scene_dict, scene_timings):
    merged_scenes = []
    current_category = None
    current_group = []
    start_time = None
    end_time = None

    # Sort scenes by their start time to maintain order
    sorted_scenes = sorted(scene_dict.keys(), key=lambda scene: scene_timings[scene][0])

    for scene in sorted_scenes:
        category = scene_dict[scene]
        scene_start, scene_end = scene_timings[scene]

        # If category changes OR if the scene is not consecutive, create a new group
        if category != current_category or (current_group and scene_start != end_time):
            if current_group:
                merged_scenes.append(((start_time, end_time), current_group, current_category))
            current_group = [scene]
            current_category = category
            start_time = scene_start
        else:
            current_group.append(scene)

        end_time = scene_end

    # Append the last group
    if current_group:
        merged_scenes.append(((start_time, end_time), current_group, current_category))

    return merged_scenes
def trim_video(input_path, output_path, trim_sections):
    try:
        video = VideoFileClip(input_path)
        print("Video Duration:", video.duration)

        # Adjust trim sections and ensure they are within bounds
        adjusted_trim_sections = [
            (max(0, start + 1.5), min(video.duration, end - 1.5))
            for start, end in trim_sections if start < video.duration and end <= video.duration
        ]
        print("Adjusted Trim Sections:", adjusted_trim_sections)

        if not adjusted_trim_sections:
            print("No valid trim sections found. Exiting.")
            return

        # Extract segments to keep
        clips = []
        last_end = 0

        for start, end in adjusted_trim_sections:
            if last_end < start:
                clips.append(video.subclip(last_end, start))
            last_end = end

        if last_end < video.duration:
            clips.append(video.subclip(last_end, video.duration))

        # Print kept clips
        kept_clips = [(c.start, c.end) for c in clips]
        print("Kept Clips:", kept_clips)

        # Concatenate the remaining parts
        if clips:
            final_video = concatenate_videoclips(clips)
            final_video.write_videofile(output_path, codec='libx264', fps=video.fps or 30)

            for clip in clips:
                clip.close()
            final_video.close()

        video.close()
        print("Video trimming successful! Saved as:", output_path)

    except Exception as e:
        print("Error:", str(e))
def time_to_seconds(time_str):
    """Convert HH:MM:SS.sss to total seconds."""
    t = datetime.strptime(time_str, "%H:%M:%S.%f")
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
def create_video_versions(original_path, base_filename):
    """
    Dummy function to create different versions of the uploaded video based on 
    fight and song removal options.
    
    Args:
        original_path: Path to the original video file
        base_filename: Base filename to use for the created versions
    
    Returns:
        Dictionary containing paths to different video versions
    """
    # Extract the extension from the original path
    extension = original_path.rsplit('.', 1)[1].lower()
    original_filename = os.path.basename(original_path)
     #detect scenes - save in folder
     # ✅ Run the pipeline
    video_path = original_path
    output_folder = "audio_clips"

    print("🎬 Detecting scenes...")
    scene_list, interval = detect_scenes(video_path, threshold=20.0, min_scene_length=3)

    print(f"\n🔊 Extracting audio from {len(scene_list)} scenes...")
    extracted_audio = extract_audio_from_scenes(video_path, scene_list, output_folder)

    print(f"\n✅ Extracted {len(extracted_audio)} audio clips.")

    # ✅ Run the pipeline
    
    output_folder = "frames"



    print(f"\n📸 Extracting key frames from {len(scene_list)} scenes at -second intervals...")
    extracted_frames = extract_scene_frames(video_path, scene_list, output_folder, interval=1)

    print(f"\n✅ Extracted {len(extracted_frames)} frames from {len(scene_list)} scenes.")



    #random forest classifier - gemini call
    dataset_path = 'audio_features.csv'
    model_path = 'audio_classifier.joblib'
    collect_feedback = True
    
    print(f"\n{'=' * 50}")
    print(f"AUDIO SCENE CLASSIFIER")
    print(f"{'=' * 50}")
    print(f"Using dataset: {dataset_path}")
    print(f"Using model: {model_path}")
    
    # Get folder path from user
    folder_path = "audio_clips"
    results={}
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        results = analyze_audio_folder(folder_path, dataset_path, model_path, collect_feedback)
        
        if results is None or not results:
            print("\nFolder analysis failed or no audio files were found.")
        else:
            
            # Show detailed results
            print("\nDetailed results:")
            for result in results:
                print(f"{result['file']} - {result['prediction']}")
                
            print("\nAnalysis completed successfully.")
            
            results = {entry['file']: entry['prediction'].strip() for entry in results}

            print("RESULTS", results)

    else:
        print("Invalid folder path. Please enter a valid path.")

        
    #merge

    merged_scenes_with_timings = merge_scenes(results, interval)
    for (start_time, end_time), scenes, category in merged_scenes_with_timings:
        print(f"{start_time} - {end_time}: {scenes} -> {category}")
    # Print the merged scenes with exact timestamps
    print("interval",interval,"\n\n\n")
    print("results",results,"\n\n\n")
    fights = []
    dialogues = []
    songs = []

    for (start_time, end_time), scenes, category in merged_scenes_with_timings:
        if category == "fights":
            fights.append((start_time, end_time))
        elif category == "dialogues":
            dialogues.append((start_time, end_time))
        elif category == "songs":
            songs.append((start_time, end_time))
    

# Convert timings to seconds
    fights = [(time_to_seconds(start), time_to_seconds(end)) for start, end in fights]
    dialogues = [(time_to_seconds(start), time_to_seconds(end)) for start, end in dialogues]
    songs = [(time_to_seconds(start), time_to_seconds(end)) for start, end in songs]
    

    # Now you have structured data
    print("Fights:", fights)
    print("Dialogues:", dialogues)
    print("Songs:", songs)
    both = fights+songs
    #trim
    trim_video(original_path, f"{VIDEO_DIR}\{base_filename}_fightremoved.{extension}", fights)
    trim_video(original_path, f"{VIDEO_DIR}\{base_filename}_songsremoved.{extension}", songs)
    trim_video(original_path, f"{VIDEO_DIR}\{base_filename}_bothremoved.{extension}", both)
    #save_trimmed
    # Create versions dictionary with the specified format
    versions = {
        "original": original_filename,
        "fightRemoved": f"{base_filename}_fightremoved.{extension}",
        "songsRemoved": f"{base_filename}_songsremoved.{extension}",
        "bothRemoved": f"{base_filename}_bothremoved.{extension}",
        "haveFights": f"{base_filename}_haveFights.{extension}",        
    }
    
    # In a real implementation, you would actually create these files
    # using video processing libraries to remove fights and songs
    print(f"Would create video versions for {original_path} with fights and songs removed")
    
    return versions

csv_path="audio_features.csv"
model_path="audio_classifier.joblib"
VIDEO_DIR = "C:/Users/Swarneshwar S/Desktop/FILES/PROJECTS/DISH/videos"
#create_video_versions("C:/Users/Swarneshwar S/Desktop/FILES/PROJECTS/DISH/videos/newmovie_c0a2e18b.mp4","newmovie_c0a2e18b")

