import os
import numpy as np
import pandas as pd
import librosa
import joblib
from pydub import AudioSegment
import yt_dlp
import tempfile
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
warnings.filterwarnings('ignore')

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
    """
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
    return sorted_probs[0][0]

def download_youtube_audio(youtube_url, output_filename=None):
    """
    Download audio from YouTube video
    """
    print(f"Downloading audio from: {youtube_url}")
    
    if output_filename is None:
        # Use temp directory if no output dir specified
        output_dir = tempfile.gettempdir()
        # Create unique filename based on video ID
        video_id = youtube_url.split("watch?v=")[-1].split("&")[0]
        output_filename = os.path.join(output_dir, f"yt_{video_id}.mp3")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'extractaudio': True,
        'audioformat': 'mp3',
        'outtmpl': output_filename,
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        print(f"✓ Download complete: {output_filename}")
        return output_filename
    except Exception as e:
        print(f"× Error downloading audio: {e}")
        return None

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
            print("close prediction")
            final_prediction = resolve_close_predictions(class_probabilities, threshold, file_path)
        else:
            final_prediction = prediction[0]
            
        print(f"✓ Prediction complete: {final_prediction}")
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

def analyze_youtube_audio(youtube_url, csv_path="audio_features.csv", model_path="audio_classifier.joblib", keep_files=False):
    """
    Main function to analyze YouTube audio
    """
    print(f"\n{'=' * 50}")
    print(f"AUDIO SCENE CLASSIFIER - YOUTUBE URL")
    print(f"{'=' * 50}")
    
    temp_files = []
    
    # Step 1: Train or load the model
    if os.path.exists(model_path):
        print(f"Loading existing model: {model_path}")
        model = joblib.load(model_path)
    else:
        print(f"Training new model from: {csv_path}")
        model = train_model(csv_path, model_path)
        if model is None:
            return None
    
    # Step 2: Download audio from YouTube
    audio_file = download_youtube_audio(youtube_url)
    if audio_file is None:
        return None
    temp_files.append(audio_file)
    
    # Step 3: Convert audio to proper format
    converted_file = convert_audio(audio_file)
    if converted_file is None:
        if not keep_files:
            cleanup_temp_files(temp_files)
        return None
    temp_files.append(converted_file)
    
    # Step 4: Extract features
    features = extract_features(converted_file)
    if features is None:
        if not keep_files:
            cleanup_temp_files(temp_files)
        return None
    
    # Step 5: Make prediction
    prediction, probabilities = predict_audio_class(features, model, "/")
    
    # Step 6: Display results
    print(f"\n{'=' * 50}")
    print(f"RESULTS")
    print(f"{'=' * 50}")
    print(f"YouTube URL: {youtube_url}")
    print(f"Predicted class: {prediction}")
    print("\nClass probabilities:")
    for class_name, prob in probabilities.items():
        print(f"{class_name}: {prob:.2%}")
    
    # Step 7: Get feedback on prediction
    print("\nIs this prediction correct? (y/n): ")
    correct = input().strip().lower()
    
    model_retrained = False
    
    if correct != 'y':
        # Get the correct label
        print("\nWhat is the correct label for this audio?")
        
        # If model has classes, display them as options
        if hasattr(model, 'classes_'):
            print("Available classes:")
            for i, class_name in enumerate(model.classes_, 1):
                print(f"{i}. {class_name}")
            print(f"{len(model.classes_)+1}. Other (specify)")
            
            choice = input("\nEnter number: ").strip()
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(model.classes_):
                    correct_label = model.classes_[choice_idx]
                else:
                    correct_label = input("Enter the correct label: ").strip()
            except ValueError:
                correct_label = input("Enter the correct label: ").strip()
        else:
            correct_label = input("Enter the correct label: ").strip()
        
        # Add to dataset
        success = add_to_dataset(features, correct_label, csv_path)
        
        if success:
            # Ask if user wants to retrain the model
            print("\nWould you like to retrain the model with the updated dataset? (y/n): ")
            retrain = input().strip().lower()
            
            if retrain == 'y':
                # Retrain the model
                new_model = train_model(csv_path, model_path)
                if new_model is not None:
                    model = new_model
                    model_retrained = True
    
    # Step 8: Clean up temporary files if not keeping them
    if not keep_files:
        cleanup_temp_files(temp_files)
    
    # Step 9: Return results
    return {
        "prediction": prediction,
        "probabilities": probabilities,
        "features": features.tolist(),
        "feedback_collected": correct != 'y',
        "model_retrained": model_retrained
    }

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
            
        print("\nIs this prediction correct? (y/n): ")
        correct = input().strip().lower()
        
        if correct != 'y':
            # Get the correct label
            print("\nWhat is the correct label for this audio?")
            
            # If model has classes, display them as options
            if hasattr(model, 'classes_'):
                print("Available classes:")
                for i, class_name in enumerate(model.classes_, 1):
                    print(f"{i}. {class_name}")
                print(f"{len(model.classes_)+1}. Other (specify)")
                
                choice = input("\nEnter number: ").strip()
                try:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(model.classes_):
                        correct_label = model.classes_[choice_idx]
                    else:
                        correct_label = input("Enter the correct label: ").strip()
                except ValueError:
                    correct_label = input("Enter the correct label: ").strip()
            else:
                correct_label = input("Enter the correct label: ").strip()
            
            # Add to dataset
            add_to_dataset(features, correct_label, csv_path)
    
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

if __name__ == "__main__":
    # Default parameters
    dataset_path = 'audio_features.csv'
    model_path = 'audio_classifier.joblib'
    keep_files = False
    collect_feedback = True
    
    # Interactive mode
    print(f"\n{'=' * 50}")
    print(f"AUDIO SCENE CLASSIFIER - FIGHT/SONG/DIALOGUE")
    print(f"{'=' * 50}")
    
    # Ask if user wants to customize settings
    print("\nWould you like to use default settings or customize?")
    print("1. Use defaults")
    print("2. Customize settings")
    
    choice = input("\nEnter your choice (1/2): ").strip()
    
    if choice == "2":
        # Get custom settings
        dataset_path = input(f"\nEnter dataset path [{dataset_path}]: ").strip() or dataset_path
        model_path = input(f"Enter model path [{model_path}]: ").strip() or model_path
        
        feedback_choice = input(f"Collect feedback on predictions? (y/n) [y]: ").strip().lower() or 'y'
        collect_feedback = feedback_choice == 'y'
        
        keep_choice = input("Keep temporary files after YouTube analysis? (y/n) [n]: ").strip().lower()
        keep_files = keep_choice == 'y'
    
    # Main menu
    while True:
        print("\nWhat would you like to do?")
        print("1. Analyze a YouTube video")
        print("2. Analyze a folder of audio files")
        print("3. Train/retrain the model")
        print("4. Exit")
        
        main_choice = input("\nEnter your choice (1-4): ").strip()
        
        if main_choice == "1":
            # Analyze YouTube video
            youtube_url = input("\nEnter YouTube URL: ").strip()
            
            if "youtube.com/watch?v=" in youtube_url or "youtu.be/" in youtube_url:
                # Run the analysis
                result = analyze_youtube_audio(youtube_url, dataset_path, model_path, keep_files)
                
                if result is None:
                    print("\nAnalysis failed. Please check the errors above.")
            else:
                print("Invalid YouTube URL. Please enter a valid URL.")
                
        elif main_choice == "2":
            # Analyze folder
            folder_path = input("\nEnter folder path containing audio files: ").strip()
            
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                results = analyze_audio_folder(folder_path, dataset_path, model_path, collect_feedback)
                
                if results is None or not results:
                    print("\nFolder analysis failed or no audio files were found.")
                else:
                    # Show detailed results
                    print("\nDetailed results:")
                    for result in results:
                        print(f"{result['file']} - {result['prediction']}")
            else:
                print("Invalid folder path. Please enter a valid path.")
                
        elif main_choice == "3":
            # Train/retrain model
            print(f"\nTraining model using dataset: {dataset_path}")
            model = train_model(dataset_path, model_path)
            
            if model is not None:
                print("\nModel training completed successfully.")
            else:
                print("\nModel training failed. Please check the errors above.")
                
        elif main_choice == "4":
            # Exit
            print("\nExiting program. Goodbye!")
            break
            
        else:
            print("\nInvalid choice. Please enter a number between 1 and 4.")