import cv2
import numpy as np
import google.generativeai as genai
import os
import time
from google.api_core.exceptions import InvalidArgument
import pyaudio
import wave
from pathlib import Path
import threading
import tempfile
import queue
import webrtcvad # Added for VAD

# Configuration
API_KEY = os.environ.get("GOOGLE_API_KEY")  # Set your API key as an environment variable
TEMP_DIR = Path(tempfile.gettempdir()) / "gemini_camera_analysis"
TEMP_DIR.mkdir(exist_ok=True)

# Initialize Gemini API
genai.configure(api_key=API_KEY)

# Audio recording parameters
CHUNK_DURATION_MS = 30  # VAD requires 10, 20, or 30 ms chunks
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)  # frames per buffer
# RECORD_SECONDS = 5  # No longer needed for fixed segments
VAD_AGGRESSIVENESS = 3  # 0 (least aggressive) to 3 (most aggressive)
SILENCE_THRESHOLD_FRAMES = int(1.0 * RATE / CHUNK_SIZE) # 1 second of silence to end segment
MIN_SPEECH_FRAMES = int(0.5 * RATE / CHUNK_SIZE) # Minimum 0.5 seconds of speech

# Create queues for processing
video_queue = queue.Queue()
audio_queue = queue.Queue()
result_queue = queue.Queue()

# Function to capture video frames
def capture_video():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Display the frame
            cv2.imshow("Camera Feed", frame)
            
            # Add frame to processing queue (downsize for efficiency)
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            video_queue.put(small_frame)
            
            # Break loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.1)  # Reduce frame rate for processing
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Function to record audio using VAD
def record_audio():
    audio = pyaudio.PyAudio()
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                       rate=RATE, input=True,
                       frames_per_buffer=CHUNK_SIZE)
    
    print("Listening for speech...")
    frames_buffer = []
    silence_counter = 0
    is_speaking = False
    
    try:
        while True:
            data = stream.read(CHUNK_SIZE)
            try:
                is_speech = vad.is_speech(data, RATE)
            except Exception as vad_error:
                print(f"VAD Error: {vad_error}")
                continue # Skip this chunk

            if is_speech:
                frames_buffer.append(data)
                silence_counter = 0
                if not is_speaking:
                    print("Speech detected...")
                    is_speaking = True
            elif is_speaking:
                # Still append some silence frames after speech ends
                frames_buffer.append(data)
                silence_counter += 1
                if silence_counter > SILENCE_THRESHOLD_FRAMES:
                    print(f"Silence detected, processing segment ({len(frames_buffer) * CHUNK_DURATION_MS / 1000:.2f}s)...")
                    if len(frames_buffer) > MIN_SPEECH_FRAMES:
                        # Save temporary audio file
                        temp_audio_path = TEMP_DIR / f"audio_{time.time()}.wav"
                        wf = wave.open(str(temp_audio_path), 'wb')
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(audio.get_sample_size(FORMAT))
                        wf.setframerate(RATE)
                        wf.writeframes(b''.join(frames_buffer))
                        wf.close()
                        
                        # Add to processing queue
                        audio_queue.put(str(temp_audio_path))
                    else:
                        print("Segment too short, discarding.")
                    
                    # Reset for next segment
                    frames_buffer = []
                    silence_counter = 0
                    is_speaking = False
                    print("Listening for speech...")
            # else: # Silence while not speaking, do nothing
            #     pass 

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

# Function to process media with Gemini
def process_media():
    # Initialize Gemini model
    # Check if model exists before initializing - Added check
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
    except Exception as model_init_error:
        print(f"Error initializing Gemini model: {model_init_error}")
        print("Please ensure your API key is valid and the model name is correct.")
        # Add the result to the queue to signal the error to the display thread
        result_queue.put(f"FATAL ERROR: Could not initialize Gemini model: {model_init_error}")
        return # Stop the processing thread if model fails to load

    while True:
        try:
            # Get latest frame and audio if available
            if not video_queue.empty():
                frame = video_queue.get()
                
                # Save temporary image
                temp_img_path = TEMP_DIR / f"frame_{time.time()}.jpg"
                cv2.imwrite(str(temp_img_path), frame)
                
                # Process image with Gemini
                with open(temp_img_path, 'rb') as f:
                    image_data = f.read()
                
                # Generate content safely - Added error handling
                try:
                    response = model.generate_content([
                        "Describe what you see in this image, focusing on any text, people, or important elements.",
                        {'mime_type': 'image/jpeg', 'data': image_data} # Use dictionary format
                    ])
                    result_queue.put(f"Visual analysis: {response.text}")
                except InvalidArgument as e:
                    result_queue.put(f"API Image Error: {str(e)}")
                except Exception as e:
                    result_queue.put(f"Image Processing error: {str(e)}")
                finally:
                    # Clean up temp file
                    try: # Added try-except for unlink
                        temp_img_path.unlink()
                    except OSError as unlink_error:
                        print(f"Error deleting temp image file {temp_img_path}: {unlink_error}")
            
            # Process audio if available
            if not audio_queue.empty():
                audio_path = audio_queue.get()
                
                # Process audio with Gemini
                # Use UploadedFile API for better handling
                try:
                    print(f"Uploading audio file: {audio_path}")
                    audio_file = genai.upload_file(path=audio_path)
                    print(f"Audio file uploaded successfully: {audio_file.name}")

                    # Wait for file processing if needed (usually fast)
                    while audio_file.state.name == "PROCESSING":
                        print('.', end='')
                        time.sleep(1)
                        audio_file = genai.get_file(audio_file.name)
                    
                    if audio_file.state.name == "FAILED":
                         raise Exception(f"Audio file processing failed: {audio_file.state.name}")

                    print(f"\nGenerating content for audio: {audio_file.name}")
                    response = model.generate_content([
                        "Transcribe and summarize what is being said in this audio.",
                        audio_file
                    ])
                    
                    result_queue.put(f"Audio analysis: {response.text}")
                    
                    # Clean up uploaded file on API side
                    genai.delete_file(audio_file.name)
                    print(f"Deleted uploaded file: {audio_file.name}")

                except InvalidArgument as e:
                    result_queue.put(f"API Audio Error: {str(e)}")
                except Exception as e:
                    result_queue.put(f"Audio Processing error: {str(e)}")
                finally:
                     # Clean up local temp file
                    try: # Added try-except for unlink
                        Path(audio_path).unlink()
                        print(f"Deleted local temp file: {audio_path}")
                    except OSError as unlink_error:
                        print(f"Error deleting temp audio file {audio_path}: {unlink_error}")
            
            time.sleep(0.2)  # Slightly shorter pause 
            
        except Exception as e:
            # Catch potential errors in the main loop of process_media
            result_queue.put(f"General processing loop error: {str(e)}")
            time.sleep(1) # Pause before retrying

# Function to display results
def display_results():
    while True:
        if not result_queue.empty():
            result = result_queue.get()
            print("\n" + "="*50)
            print(result)
            print("="*50)
        
        time.sleep(0.5)

def main():
    if not API_KEY:
        print("Error: Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")
        return
    
    print("Starting Gemini Camera Analysis")
    print("Press 'q' in the camera window to quit")
    
    # Start threads
    threads = [
        threading.Thread(target=capture_video, daemon=True),
        threading.Thread(target=record_audio, daemon=True),
        threading.Thread(target=process_media, daemon=True),
        threading.Thread(target=display_results, daemon=True)
    ]
    
    for thread in threads:
        thread.start()
    
    try:
        # Keep main thread alive
        threads[0].join()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Clean up any temporary files
        for file in TEMP_DIR.glob("*"):
            try:
                file.unlink()
            except:
                pass
        try:
            TEMP_DIR.rmdir()
        except:
            pass

if __name__ == "__main__":
    main() 