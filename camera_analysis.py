import cv2
import numpy as np
import google.generativeai as genai
import os
import time
from google.api_core.exceptions import InvalidArgument
import queue
from pathlib import Path
import threading
import tempfile
from pyneuphonic import Neuphonic, TTSConfig
from pyneuphonic.player import AudioPlayer
import json

import face_recognition

client = Neuphonic('42ab0121289216df4abf58f9640c711ac2e1de42845bee6b1a619ffd082da9c2.ef521e59-89e4-4e1c-835c-2989af341bff')

sse = client.tts.SSEClient()
tts_config = TTSConfig(
    speed=1.2,
    lang_code='en',
    sampling_rate=22050,
)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

social_score = {}
social_score['blue'] = 100
social_score['red'] = 100

image_of_person  = face_recognition.load_image_file("marcus.jpg")
marcus_encoding = face_recognition.face_encodings(image_of_person)[0]

known_face_encodings = [marcus_encoding]
konwn_face_names = ["Marcus"]



BLUE_ON_SCREEN = False
RED_ON_SCREEN = False

# Configuration
API_KEY = os.environ.get("GOOGLE_API_KEY")  # Set your API key as an environment variable
TEMP_DIR = Path(tempfile.gettempdir()) / "gemini_camera_analysis"
TEMP_DIR.mkdir(exist_ok=True)

# Initialize Gemini API
genai.configure(api_key=API_KEY)

# Create queues for processing
video_queue = queue.Queue()
result_queue = queue.Queue()

# Function to capture video frames
def capture_video(display_queue, video_queue, stop_event):
    # Removed camera init
    try:
        while not stop_event.is_set():
            try:
                frame = display_queue.get(timeout=2.5) # Wait briefly for a frame
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

# Function to process media with Gemini
def process_media(video_queue, result_queue, stop_event):
    # Initialize Gemini model
    model = None # Initialize model to None
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
    except Exception as model_init_error:
        print(f"Error initializing Gemini model: {model_init_error}")
        result_queue.put(f"FATAL ERROR: Could not initialize Gemini model: {model_init_error}")#
        return # Stop thread

    last_process_time = time.time() - 5.0 # Process first frame immediately

    while not stop_event.is_set(): # Check stop event
        current_time = time.time()
        frame_to_process = None
        processed_frame_this_cycle = False

        # Check if 5 seconds have passed
        if current_time - last_process_time >= 5.0:
            latest_frame = None
            try:
                # Consume all frames currently in the queue to get the latest one
                while True:
                    latest_frame = video_queue.get_nowait()
            except queue.Empty:
                pass # Keep the latest frame obtained (if any)

            if latest_frame is not None:
                frame_to_process = latest_frame
                last_process_time = current_time # Update time only when processing

        # Process the selected frame if available
        if frame_to_process is not None:
            processed_frame_this_cycle = True
            try:
                # --- Process frame ---
                temp_img_path = TEMP_DIR / f"frame_{time.time()}.jpg"
                cv2.imwrite(str(temp_img_path), frame_to_process) # Use frame_to_process
                with open(temp_img_path, 'rb') as f:
                    image_data = f.read()
                try:
                    print(f"Processing frame captured at ~{time.strftime('%H:%M:%S', time.localtime(last_process_time))}")
                    prompt = """Generate a JSON object containing exactly two keys: "name" and "action".

- The value for the "name" key should be a string representing the main person or object detected in the image. If multiple are present, focus on the most prominent one.
- Use the primary of what the person is wearing for now e.g blue, red
- The value for the "action" key should be a string describing what the identified person or object is doing.
- There is two person you need to identify and describe their actions within the frame. A is in red and B is in blue.

Provide only the JSON object as the output, with no additional text before or after it.

Example:
[
    {
        "name": "blue",
        "action": "drink"
    },
    {
        "name": "red",
        "action": "using phone"
    }
]

This is an example if there is two people on camera
The only actions you can describe are:
- drinking
- using phone
- looking sad

If their action do not match any of the above, describe it as "HAPPY"

Now, generate the JSON based on the provided image.
"""
                    response = model.generate_content([
                        prompt,
                        {'mime_type': 'image/jpeg', 'data': image_data}
                    ])
                    result_queue.put(f"{response.text}")
                except InvalidArgument as e:
                    result_queue.put(f"API Image Error: {str(e)}")
                except Exception as e:
                    result_queue.put(f"Image Processing error: {str(e)}")
                finally:
                    try:
                        temp_img_path.unlink()
                    except OSError as unlink_error:
                        print(f"Error deleting temp image file {temp_img_path}: {unlink_error}")
                # --- End frame processing ---
            except Exception as e:
                result_queue.put(f"General frame processing error: {str(e)}")

        # Sleep briefly if no frame was processed in this cycle to avoid busy-waiting
        if not processed_frame_this_cycle:
            time.sleep(0.1) # Check again in 100ms

    print("Process media thread finished.")

def talk(text):
    try:
        if(text):
            print("HERERE")
            text = text.strip("```json\n").strip("```")
            data = json.loads(text)
            message = ""
            for entry in data:
                name = entry.get("name")
                action = entry.get("action")
                if(name == "blue" or name == "red"):
                    if(action == "drinking"):
                        message = "stop drinking!"
                        social_score[name] -= 10
                    if(action == "using phone"):
                        message = "stop using your phone!"
                        social_score[name] -= 10
                    if(action == "looking sad"):
                        message = "stop looking sad"
                        social_score[name] -= 10
                    if(name == "blue"):
                        BLUE_ON_SCREEN = True
                    if(name == "red"):
                        RED_ON_SCREEN = True
            with AudioPlayer(sampling_rate=22050) as player:
                    response = sse.send(message, tts_config=tts_config)
                    player.play(response)

    except Exception as e:
        print(str(e))
    

# Function to display results
def display_results(result_queue, stop_event):
    while not stop_event.is_set(): # Check stop event
        try:
            result = result_queue.get(timeout=1) # Check queue with timeout
            print("\n" + "="*50)
            print(result)

            
            threading.Thread(target=talk, args=(result,), daemon=True).start()

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

    
def main():
    if not API_KEY:
        print("Error: Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")
        return
    
    #train_recognizer()
    #recognizer.read('face_trainer.yml')
    
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
        threading.Thread(target=process_media, args=(video_queue, result_queue, stop_event), daemon=True),
        threading.Thread(target=display_results, args=(result_queue, stop_event), daemon=True),
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
                #cv2.imshow("Camera Feed", frame)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                # Loop through each face found
                
                for (x, y, w, h) in faces:
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    # Crop the detected face from the original frame
                    face = frame[y:y + h, x:x + w]

                    # Optional: Draw something above the head (label)
                    label_text = str(social_score["blue"])
                    label_y = max(20, y - 20)  # Ensure it's not off-screen

                    # Draw a filled rectangle for label background (optional)
                    cv2.rectangle(frame, (x, label_y - 20), (x + w, label_y), (0, 0, 0), -1)

                    # Draw label (text)
                    cv2.putText(frame, label_text, (x + 5, label_y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                cv2.imshow("Face", frame)
                    
                    

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
            thread.join(timeout=2.5) # Add timeout to join
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