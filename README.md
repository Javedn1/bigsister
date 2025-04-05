# Gemini Camera Analysis

This Python application uses the Gemini 2.0 Flash API to analyze video and audio from your camera, outputting text descriptions of what is seen and heard.

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
# You might need to install system libraries for PyAudio and webrtcvad depending on your OS.
# For example, on Debian/Ubuntu:
# sudo apt-get update
# sudo apt-get install portaudio19-dev python3-pyaudio
```

2. Set your Google API key as an environment variable:
```
export GOOGLE_API_KEY="your-api-key-here"
```

3. Run the application:
```
python camera_analysis.py
```

## Usage

- The application will open your camera and start recording audio in 5-second segments.
- Video frames and audio will be processed through the Gemini 2.0 Flash API.
- Text analysis results will be displayed in the terminal.
- Press 'q' in the camera window to quit.

## Requirements

- Python 3.7+
- A webcam
- A microphone
- Google API key with access to Gemini 2.0 Flash

## Note

The application creates temporary files for processing that are automatically cleaned up when the program exits. 