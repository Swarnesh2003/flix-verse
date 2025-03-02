from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import uuid
import latest
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Video directory path
VIDEO_DIR = "C:/Users/Swarneshwar S/Desktop/FILES/PROJECTS/DISH/videos"
# Ensure directory exists
os.makedirs(VIDEO_DIR, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Mock user database
users = {
    "user1": "password1",
    "admin": "admin123",
    "test": "test123"
}

# Mock video database with information about different versions
videos = {
    1: {
        "id": 1,
        "title": "Action Movie 1", 
        "base_filename": "movie1",
        "genre": "action",
        "hasSongs": True, 
        "hasFights": True,
        # Path to different versions based on preferences
        "versions": {
            "original": "movie1.mp4",
            "fightRemoved": "movie1_fightremoved.mp4",
            "songsRemoved": "movie1_songsremoved.mp4",
            "bothRemoved": "movie1_bothremoved.mp4"
        }
    },
    2: {
        "id": 2,
        "title": "Comedy Show 2", 
        "base_filename": "movie2",
        "genre": "comedy",
        "hasSongs": True, 
        "hasFights": False,
        "versions": {
            "original": "movie2.mp4",
            "songsRemoved": "movie2_songsremoved.mp4"
        }
    },
    3: {
        "id": 3,
        "title": "Drama Series 3", 
        "base_filename": "movie3",
        "genre": "drama",
        "hasSongs": True, 
        "hasFights": False,
        "versions": {
            "original": "movie3.mp4",
            "songsRemoved": "movie3_songsremoved.mp4"
        }
    },
    4: {
        "id": 4,
        "title": "Sci-Fi Epic 4", 
        "base_filename": "movie4",
        "genre": "scifi",
        "hasSongs": False, 
        "hasFights": True,
        "versions": {
            "original": "movie4.mp4",
            "fightRemoved": "movie4_fightremoved.mp4"
        }
    },
    5: {
        "id": 5,
        "title": "Romance Film 5", 
        "base_filename": "movie5",
        "genre": "romance",
        "hasSongs": True, 
        "hasFights": False,
        "versions": {
            "original": "movie5.mp4",
            "songsRemoved": "movie5_songsremoved.mp4"
        }
    },
    6: {
        "id": 6,
        "title": "Guna", 
        "base_filename": "guna_cc84b931",
        "genre": "romance",
        "hasSongs": True, 
        "hasFights": True,
        "versions": {
            "original": "guna_cc84b931.mp4",
            "fightRemoved": "guna_cc84b931_fightremoved.mp4",
            "songsRemoved": "guna_cc84b931_songsremoved.mp4"
        }
    }
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_next_id():
    return max(videos.keys()) + 1 if videos else 1

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if username in users and users[username] == password:
        # Add user role for admin access
        role = "admin" if username == "admin" else "user"
        return jsonify({
            "success": True, 
            "message": "Login successful",
            "role": role
        }), 200
    else:
        return jsonify({"success": False, "message": "Invalid credentials"}), 401

@app.route('/api/videos', methods=['GET'])
def get_videos():
    # Return list of videos for the frontend
    videos_list = []
    for video_id, video_data in videos.items():
        videos_list.append({
            "id": video_id,
            "title": video_data["title"],
            "genre": video_data["genre"],
            "thumbnail": f"/api/placeholder/300/169",  # Placeholder for thumbnails
            "hasSongs": video_data["hasSongs"],
            "hasFights": video_data["hasFights"]
        })
    
    return jsonify({"videos": videos_list}), 200

@app.route('/api/video', methods=['POST'])
def get_video():
    data = request.json
    video_id = data.get('videoId')
    preferences = data.get('preferences', {})
    
    # Validate video_id
    try:
        video_id = int(video_id)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid video ID"}), 400
    
    # Check if video exists
    if video_id not in videos:
        return jsonify({"error": "Video not found"}), 404
    
    video = videos[video_id]
    
    # Get user preferences
    skip_songs = preferences.get('skipSongs', False)
    skip_fights = preferences.get('skipFights', False)
    
    # Determine which version to serve based on preferences
    version = "original"
    version_description = "Original"
    
    if skip_songs and skip_fights and video["hasSongs"] and video["hasFights"]:
        version = "bothRemoved"
        version_description = "Songs and Fights Removed"
    elif skip_songs and video["hasSongs"]:
        version = "songsRemoved"
        version_description = "Songs Removed"
    elif skip_fights and video["hasFights"]:
        version = "fightRemoved"
        version_description = "Fights Removed"
    
    # Get the appropriate video filename
    filename = video["versions"].get(version, video["versions"]["original"])
    
    # Create a URL that points to our stream endpoint
    stream_url = f"/api/stream/{filename}"
    
    response = {
        "id": video_id,
        "title": video["title"],
        "path": stream_url,  # URL for the frontend to use
        "version": version_description,
        "baseFilename": video["base_filename"],
        "preferences": {
            "skipSongs": skip_songs and video["hasSongs"],
            "skipFights": skip_fights and video["hasFights"]
        }
    }
    
    return jsonify(response), 200

@app.route('/api/admin/upload', methods=['POST'])
def upload_video():
    """Handle video uploads from admin dashboard"""
    # Check if the post request has the file part
    if 'videoFile' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['videoFile']
    
    # If user does not select file, browser also submits an empty part without filename
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Supported formats: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    # Get video metadata from form data
    title = request.form.get('title', 'Untitled')
    genre = request.form.get('genre', 'other')
    has_songs = request.form.get('hasSongs') == 'true'
    has_fights = request.form.get('hasFights') == 'true'
    
    # Generate a unique base filename
    base_filename = f"{secure_filename(title.lower().replace(' ', '_'))}_{uuid.uuid4().hex[:8]}"
    filename = f"{base_filename}.{file.filename.rsplit('.', 1)[1].lower()}"
    
    # Save the file
    file_path = os.path.join(VIDEO_DIR, filename)
    file.save(file_path)
    
    # Process video to create different versions
    versions = latest.create_video_versions(file_path, base_filename)
    
    # Add video to the database
    new_id = get_next_id()
    videos[new_id] = {
        "id": new_id,
        "title": title,
        "base_filename": base_filename,
        "genre": genre,
        "hasSongs": has_songs,
        "hasFights": has_fights,
        "versions": versions
    }
    print(videos[new_id])
    return jsonify({
        "success": True,
        "message": "Video uploaded successfully",
        "video": {
            "id": new_id,
            "title": title,
            "genre": genre,
            "hasSongs": has_songs,
            "hasFights": has_fights,
            "filename": filename,
            "versions": versions
        }
    }), 201



@app.route('/api/admin/videos', methods=['GET'])
def get_admin_videos():
    """Get detailed video information for admin dashboard"""
    admin_videos = []
    for video_id, video_data in videos.items():
        admin_videos.append({
            "id": video_id,
            "title": video_data["title"],
            "genre": video_data["genre"],
            "hasSongs": video_data["hasSongs"],
            "hasFights": video_data["hasFights"],
            "baseFilename": video_data["base_filename"],
            "versions": list(video_data["versions"].keys())
        })
    
    return jsonify({"videos": admin_videos}), 200

@app.route('/api/admin/upload-version', methods=['POST'])
def upload_version():
    """Upload an alternative version of an existing video"""
    if 'videoFile' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['videoFile']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Supported formats: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    # Get metadata from form data
    video_id = request.form.get('videoId')
    version_type = request.form.get('versionType')
    
    # Validate data
    try:
        video_id = int(video_id)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid video ID"}), 400
    
    if video_id not in videos:
        return jsonify({"error": "Video not found"}), 404
        
    if not version_type or version_type not in ['fightRemoved', 'songsRemoved', 'bothRemoved']:
        return jsonify({"error": "Invalid version type"}), 400
        
    # Get the video data
    video = videos[video_id]
    
    # Create filename for the version
    extension = file.filename.rsplit('.', 1)[1].lower()
    version_filename = f"{video['base_filename']}_{version_type}.{extension}"
    
    # Save the file
    file_path = os.path.join(VIDEO_DIR, version_filename)
    file.save(file_path)
    
    # Update the versions dict
    video['versions'][version_type] = version_filename
    
    return jsonify({
        "success": True,
        "message": f"Version '{version_type}' uploaded successfully",
        "video": {
            "id": video_id,
            "title": video["title"],
            "versions": list(video["versions"].keys())
        }
    }), 201

@app.route('/api/stream/<path:filename>', methods=['GET'])
def stream_video(filename):
    """Stream video file from the server to the client."""
    return send_from_directory(VIDEO_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True)