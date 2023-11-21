from flask import Flask, request, send_file
from moviepy.editor import VideoFileClip

import os

app = Flask(__name__)

# Create a route to extract the audio from the video
@app.route('/extract-audio', methods=['POST'])
def extract_audio():
    if 'video' not in request.files:
        return 'No video uploaded', 400
    video_file = request.files['video']
    if video_file.filename == '':
        return 'No video uploaded', 400
    
    # Save the video file to disk
    video_path = os.path.join('/tmp', video_file.filename)
    audio_path = os.path.splitext(video_path)[0] + '.mp3'
    
    video_file.save(video_path)
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)

    return send_file(audio_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

