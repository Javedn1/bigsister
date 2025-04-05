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
from pyneuphonic import Neuphonic, TTSConfig
from pyneuphonic.player import AudioPlayer

#remembering faces

client = Neuphonic('42ab0121289216df4abf58f9640c711ac2e1de42845bee6b1a619ffd082da9c2.ef521e59-89e4-4e1c-835c-2989af341bff')
sse = client.tts.SSEClient()

tts_config = TTSConfig(
    lang_code='en',
    sampling_rate=22050,
)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Initialize face recognizer (LBPH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

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
def capture_video(display_queue, video_queue, stop_event):
    # Removed camera init
    try:
        while not stop_event.is_set():
            try:
                frame = display_queue.get(timeout=0.5) # Wait briefly for a frame
                if frame is None: # Check for sentinel value
                    break
                
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                video_queue.put(small_frame)

            except queue.Empty:
                # If queue is empty, loop back and check stop_event again
                continue

    finally:
        print("Capture video thread finished.")
        # Removed cap.release()

# Function to record audio using VAD
def record_audio(audio_queue, stop_event):
    audio = pyaudio.PyAudio()
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    
    try: # Added try-finally for resource cleanup
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                           rate=RATE, input=True,
                           frames_per_buffer=CHUNK_SIZE)
        
        print("Listening for speech...")
        frames_buffer = []
        silence_counter = 0
        is_speaking = False
        
        while not stop_event.is_set(): # Check stop event
            try:
                # Use non-blocking read with timeout to allow checking stop_event
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            except IOError as e:
                 # Handle input overflow gracefully if it occurs
                 if e.errno == pyaudio.paInputOverflowed:
                     print("Warning: Audio input overflowed. Skipping chunk.")
                     data = None
                 else:
                     raise # Re-raise other IOErrors

            if data is None: # Skip if overflow occurred
                time.sleep(0.01) # Short sleep to prevent busy-waiting
                continue

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
                frames_buffer.append(data)
                silence_counter += 1
                if silence_counter > SILENCE_THRESHOLD_FRAMES:
                    print(f"Silence detected, processing segment ({len(frames_buffer) * CHUNK_DURATION_MS / 1000:.2f}s)...")
                    if len(frames_buffer) > MIN_SPEECH_FRAMES:
                        temp_audio_path = TEMP_DIR / f"audio_{time.time()}.wav"
                        wf = wave.open(str(temp_audio_path), 'wb')
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(audio.get_sample_size(FORMAT))
                        wf.setframerate(RATE)
                        wf.writeframes(b''.join(frames_buffer))
                        wf.close()
                        audio_queue.put(str(temp_audio_path))
                    else:
                        print("Segment too short, discarding.")
                    frames_buffer = []
                    silence_counter = 0
                    is_speaking = False
                    print("Listening for speech...")
            
            # Add a small sleep if VAD logic didn't run to prevent busy-waiting
            # (This might not be strictly necessary with stream.read timeout)
            # time.sleep(0.01) 

    finally:
        print("Record audio thread finished.")
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        audio.terminate()

# Function to process media with Gemini
def process_media(video_queue, audio_queue, result_queue, stop_event):
    # Initialize Gemini model
    model = None # Initialize model to None
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
    except Exception as model_init_error:
        print(f"Error initializing Gemini model: {model_init_error}")
        result_queue.put(f"FATAL ERROR: Could not initialize Gemini model: {model_init_error}")#
        return # Stop thread

    while not stop_event.is_set(): # Check stop event
        processed_something = False
        try:
            # Check video queue with timeout
            try:
                frame = video_queue.get(timeout=0.1) 
                # --- Process frame (mostly unchanged) ---
                temp_img_path = TEMP_DIR / f"frame_{time.time()}.jpg"
                cv2.imwrite(str(temp_img_path), frame)
                with open(temp_img_path, 'rb') as f:
                    image_data = f.read()
                try:
                    response = model.generate_content([
                        "DO NOT RESPOND WITH VISUAL ANALYSIS, DESCRIBE WHAT YOU SEE, don't describe any background, just describe actions",
                        {'mime_type': 'image/jpeg', 'data': image_data}
                    ])
                    result_queue.put(f"Visual analysis: {response.text}")
                except InvalidArgument as e:
                    result_queue.put(f"API Image Error: {str(e)}")
                except Exception as e:
                    result_queue.put(f"Image Processing error: {str(e)}")
                finally:
                    try: 
                        temp_img_path.unlink()
                    except OSError as unlink_error:
                        print(f"Error deleting temp image file {temp_img_path}: {unlink_error}")
                processed_something = True
                # --- End frame processing ---
            except queue.Empty:
                pass # No frame, continue to check audio
            
            # Check audio queue with timeout
            try:
                audio_path = audio_queue.get(timeout=0.1) 
                # --- Process audio (mostly unchanged) ---
                try:
                    print(f"Uploading audio file: {audio_path}")
                    audio_file = genai.upload_file(path=audio_path)
                    # ... (rest of audio processing, file state check, generate, delete) ...
                    print(f"Audio file uploaded successfully: {audio_file.name}")
                    while audio_file.state.name == "PROCESSING":
                        # Check stop event while waiting for upload processing
                        if stop_event.is_set(): 
                            print("Stop event during audio upload processing.")
                            # Attempt to clean up the uploaded file if possible
                            try: genai.delete_file(audio_file.name)
                            except Exception: pass 
                            break # Exit inner loop
                        print('.', end='')
                        time.sleep(1)
                        try: # Add try/except for get_file during shutdown
                           audio_file = genai.get_file(audio_file.name)
                        except Exception as get_file_err:
                            print(f"\nError checking audio file status during wait: {get_file_err}")
                            break # Exit inner loop
                    
                    if stop_event.is_set(): break # Exit outer loop if stopped

                    if audio_file.state.name == "FAILED":
                         raise Exception(f"Audio file processing failed: {audio_file.state.name}")

                    print(f"\nGenerating content for audio: {audio_file.name}")
                    response = model.generate_content([
                        "Transcribe and summarize what is being said in this audio.",
                        audio_file
                    ])
                    result_queue.put(f"Audio analysis: {response.text}")
                    genai.delete_file(audio_file.name)
                    print(f"Deleted uploaded file: {audio_file.name}")
                except InvalidArgument as e:
                    result_queue.put(f"API Audio Error: {str(e)}")
                except Exception as e:
                    result_queue.put(f"Audio Processing error: {str(e)}")
                finally:
                    try: 
                        Path(audio_path).unlink()
                        print(f"Deleted local temp file: {audio_path}")
                    except OSError as unlink_error:
                        print(f"Error deleting temp audio file {audio_path}: {unlink_error}")
                processed_something = True
                # --- End audio processing ---
            except queue.Empty:
                pass # No audio, continue
            
            # If neither queue had data, sleep briefly to prevent busy-waiting
            if not processed_something:
                time.sleep(0.1)
            
        except Exception as e:
            result_queue.put(f"General processing loop error: {str(e)}")
            time.sleep(1) 
    
    print("Process media thread finished.")

# Function to display results
def display_results(result_queue, stop_event):
    while not stop_event.is_set(): # Check stop event
        try:
            result = result_queue.get(timeout=0.5) # Check queue with timeout
            print("\n" + "="*50)
            print(result)
            with AudioPlayer(sampling_rate=22050) as player:
                response = sse.send(result, tts_config=tts_config)
                player.play(response)
            print("="*50)
        except queue.Empty:
            # If queue is empty and stop event is set, exit the loop
            if stop_event.is_set():
                break
            else:
                continue # Otherwise, keep waiting
        except Exception as e:
            print(f"Error displaying result: {e}") # Handle potential errors during print

    # Process any remaining items in the queue after stop signal
    print("Display results thread draining final queue...")
    while not result_queue.empty():
        try:
            result = result_queue.get_nowait()
            print("\n" + "="*50 + " (Final)")
            print(result)
            print("="*50)
        except queue.Empty:
            break
        except Exception as e:
            print(f"Error displaying final result: {e}")
    print("Display results thread finished.")


def train_recognizer():
    faces = []
    labels = []
    label_counter = 0
    # Loop through all images in the 'faces' directory
    for filename in os.listdir('faces'):
        if filename.endswith('.jpg'):
            img_path = os.path.join('faces', filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            label = int(filename.split('_')[0])  # Extract label (ID) from filename (e.g., 1_face1.jpg)
            faces.append(img)
            labels.append(label)
    # Train the recognizer
    recognizer.train(faces, np.array(labels))
    recognizer.save('face_trainer.yml')  # Save the trained model for later use
    print('Recognizer trained and saved.')

def recognize_face(frame, recognizer):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        id_, confidence = recognizer.predict(roi_gray)

        if confidence < 100:
            label = f"Person {id_} ({round(100 - confidence, 2)}%)"
        else:
            label = "Unknown"
        
        # Draw rectangle around face and label it
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    return frame

def main():
    if not API_KEY:
        print("Error: Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")
        return
    
    train_recognizer()
    recognizer.read('face_trainer.yml')
    
    print("Starting Gemini Camera Analysis")
    print("Press 'q' in the camera window to quit (or Ctrl+C in terminal)")

    # Create stop event and queues
    stop_event = threading.Event()
    display_queue = queue.Queue(maxsize=1) # Queue for frames to pass to capture_video thread
    
    # Open Camera in main thread
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Start threads
    threads = [
        threading.Thread(target=capture_video, args=(display_queue, video_queue, stop_event), daemon=True),
        threading.Thread(target=record_audio, args=(audio_queue, stop_event), daemon=True),
        threading.Thread(target=process_media, args=(video_queue, audio_queue, result_queue, stop_event), daemon=True),
        threading.Thread(target=display_results, args=(result_queue, stop_event), daemon=True)
    ]
    
    for thread in threads:
        thread.start()
    
    try:
        # Main loop: Read frame, display, pass to thread, check key
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera. Exiting.")
                break

            # Display the frame (Main Thread)
            try:
                cv2.imshow("Camera Feed", frame)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                # Loop through each face found
                """
                for (x, y, w, h) in faces:
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    # Crop the detected face from the original frame
                    face = frame[y:y + h, x:x + w]

                    # Optional: Draw something above the head (label)
                    label_text = "-1000 social credit"
                    label_y = max(20, y - 20)  # Ensure it's not off-screen

                    # Draw a filled rectangle for label background (optional)
                    cv2.rectangle(frame, (x, label_y - 20), (x + w, label_y), (0, 0, 0), -1)

                    # Draw label (text)
                    cv2.putText(frame, label_text, (x + 5, label_y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    cv2.imshow('Face', face)
                    """

            except cv2.error as e:
                # Don't crash if display fails, but log it
                print(f"Warning: cv2.imshow error: {e}")
                # Consider stopping if display consistently fails?

            # Put frame in queue for capture_video thread (if queue not full)
            try:
                display_queue.put_nowait(frame)
            except queue.Full:
                # This shouldn't happen often with maxsize=1 if capture_video is running
                # If it does, maybe drop frame or use put with timeout
                # print("Warning: Display queue full, dropping frame for capture thread.")
                pass

            # Check for 'q' key press (Main Thread)
            key_pressed = -1
            try:
                # WaitKey needs a small delay to process window events
                key_pressed = cv2.waitKey(1) 
            except cv2.error as e:
                print(f"Warning: cv2.waitKey error: {e}")
                # If waitKey fails, we might not be able to quit with 'q'

            if key_pressed & 0xFF == ord('q'):
                print("'q' pressed, initiating shutdown...")
                break # Exit main loop
                
    except KeyboardInterrupt:
        print("\nCtrl+C detected, initiating shutdown...")
    finally:
        # Signal threads to stop
        print("Setting stop event...")
        stop_event.set()

        # Put a sentinel value in display_queue to unblock capture_video thread get()
        try:
            display_queue.put_nowait(None)
        except queue.Full:
            pass # If full, thread is likely blocked elsewhere or already stopping
        
        # Release camera and destroy window (Main Thread)
        print("Releasing camera...")
        cap.release()
        print("Destroying OpenCV windows...")
        try:
            cv2.destroyAllWindows()
        except cv2.error as e:
             print(f"Warning: cv2.destroyAllWindows error: {e}")
        
        # Wait for all threads to finish
        print("Waiting for threads to finish...")
        for thread in threads:
            thread.join(timeout=5.0) # Add timeout to join
            if thread.is_alive():
                 print(f"Warning: Thread {thread.name} did not finish cleanly.")

        # Clean up any temporary files
        print("Cleaning up temporary files...")
        # Check if TEMP_DIR exists before cleaning up
        if TEMP_DIR.exists():
            for file in TEMP_DIR.glob("*"):
                try:
                    file.unlink()
                except OSError as e:
                    print(f"Error removing temp file {file}: {e}") 
            try:
                TEMP_DIR.rmdir()
            except OSError as e:
                 print(f"Error removing temp dir {TEMP_DIR}: {e}")
        else:
            print(f"Temporary directory {TEMP_DIR} not found, skipping cleanup.")
            
        print("Application finished.")

if __name__ == "__main__":
    main() 