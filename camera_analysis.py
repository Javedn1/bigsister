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


import pygame
pygame.mixer.init()
ding = pygame.mixer.Sound("ding.wav")
import time
from playsound import playsound


client = Neuphonic('42ab0121289216df4abf58f9640c711ac2e1de42845bee6b1a619ffd082da9c2.ef521e59-89e4-4e1c-835c-2989af341bff')

sse = client.tts.SSEClient()
tts_config = TTSConfig(
    speed=1.2,
    lang_code='en',
    sampling_rate=22050,
)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

social_score = {}

# Initialize with just the score
STARTING_SCORE = 500
for i in range(10):
    social_score[f"Person {i+1}"] = STARTING_SCORE # Use Person 1, Person 2 etc.


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
- Use the name above the person's head for the name, it should be like Person x, where x is an integer, do not include anything else
- The value for the "action" key should be a string describing what the identified person or object is doing.
- There is two person you need to identify and describe their actions within the frame. A is in red and B is in blue.

Provide only the JSON object as the output, with no additional text before or after it.

Example:
[
    {
        "name": "Person x",
        "action": "drinking water"
    },
    {
        "name": "Person y",
        "action": "using phone"
    }
]

This is an example if there is two people on camera
The only actions you can describe are:
- drinking water
- using phone
- looking sad
- smoking
- looking happy
- drinking coke

**if the person is drinking from a black bottle, describe it as "drinking coke"**
If their action do not match any of the above, describe it as "neutral"

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
            text = text.strip("```json\n").strip("```")
            data = json.loads(text)
            message = ""
            for entry in data:
                name = entry.get("name")
                action = entry.get("action")

                # --- Update score directly --- (No status here)
                if name in social_score: # Check if person exists
                    if(action == "drinking water"):
                        message = "keep drinking!"
                        social_score[name] += 100
                    elif(action == "using phone"):
                        message = "stop using your phone!"
                        social_score[name] -= 100
                    elif(action == "looking sad"):
                        message = "stop looking sad!"
                        social_score[name] -= 100
                    elif(action == "smoking"):
                        message = "stop smoking!"
                        social_score[name] -= 100
                    elif(action == "looking happy"):
                        message = "You have a nice smile!"
                        social_score[name] += 100
                    elif(action == "drinking coke"):
                        message = "stop drinking coke!"
                        social_score[name] -= 100
                    # Ensure score doesn't go below 0 (optional)
                    # social_score[name] = max(0, social_score[name])
                else:
                     print(f"Warning: Person '{name}' from Gemini not found in social_score.")
                     continue # Skip processing this entry if name is invalid
                # --- End score update ---

                # Play TTS message if one was set
                if message:
                    with AudioPlayer(sampling_rate=22050) as player:
                        response = sse.send(message, tts_config=tts_config)
                        player.play(response)
                message = "" # Reset message for next entry

    except json.JSONDecodeError as json_err:
        print(f"Error decoding JSON in talk function: {json_err}")
        print(f"Received text: {text}")
    except KeyError as key_err:
         print(f"KeyError in talk function (likely accessing social_score['{name}']): {key_err}")
    except Exception as e:
        print(f"General error in talk function: {str(e)}")


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
    
    # --- Overlay Setup ---
    overlay_images = {} # Dictionary to hold resized overlay images
    overlay_w, overlay_h = 60, 60 # Fixed overlay size
    icon_paths = {
        'diamond': 'assets/diamond.png',
        'gold': 'assets/gold.png',
        'silver': 'assets/silver.png'
    }

    for key, path in icon_paths.items():
        try:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: Could not load overlay image '{path}' for '{key}'. Skipping this icon.")
                overlay_images[key] = None # Store None if loading failed
            else:
                resized_img = cv2.resize(img, (overlay_w, overlay_h), interpolation=cv2.INTER_AREA)
                overlay_images[key] = resized_img
                print(f"Successfully loaded and resized '{path}' for '{key}'.")
        except Exception as e:
            print(f"Warning: Error loading/resizing overlay image '{path}' for '{key}': {e}. Skipping this icon.")
            overlay_images[key] = None # Store None on error

    # Check if at least one overlay loaded successfully
    any_overlay_loaded = any(img is not None for img in overlay_images.values())
    if not any_overlay_loaded:
        print("Warning: No overlay icons could be loaded. Overlay feature disabled.")

    # --- End Overlay Setup ---

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

    previous_scores = {} # Dictionary to store scores from the previous frame check

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
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_h, frame_w = frame.shape[:2]

                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                # Keep track of persons seen this frame to update previous_scores later
                persons_seen_this_frame = set()

                for idx, (x, y, w, h) in enumerate(faces):
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    person_id = f"Person {idx+1}"
                    persons_seen_this_frame.add(person_id)

                    # Get current score, defaulting if necessary
                    current_score = social_score.get(person_id, STARTING_SCORE)

                    # Get previous score, defaulting if first time seeing this person
                    prev_score = previous_scores.get(person_id, STARTING_SCORE)

                    # --- Play ding only on threshold crossing --- 
                    play_ding_sound = False
                    if current_score <= 300 and prev_score > 300:
                        print(f"{person_id} crossed below 300 threshold.")
                        play_ding_sound = True
                    elif current_score >= 700 and prev_score < 700:
                        print(f"{person_id} crossed above 700 threshold.")
                        play_ding_sound = True

                    if play_ding_sound:
                        try:
                            ding.play(maxtime=200) # Play the ding
                        except Exception as ding_err:
                             print(f"Error playing ding sound: {ding_err}")
                    # --- End ding logic ---

                    # Draw label text
                    label_text = f"{person_id}: {current_score} BS cred"
                    label_y = max(20, y - 20)
                    cv2.rectangle(frame, (x, label_y - 20), (x + w, label_y), (0, 0, 0), -1)
                    cv2.putText(frame, label_text, (x + 5, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # --- Conditional image overlay ---
                    if any_overlay_loaded:
                        # Determine which icon key to use based on score
                        icon_key = 'gold' # Default
                        if current_score <= 300:
                            icon_key = 'silver'
                        elif current_score >= 700:
                            icon_key = 'diamond'

                                            
                        # Get the corresponding pre-resized image
                        selected_overlay = overlay_images.get(icon_key)

                        if selected_overlay is not None:
                            try:
                                # Calculate position (same as before)
                                offset_x = 5
                                overlay_x_start = x + w + offset_x
                                overlay_y_start = y
                                overlay_x_end = overlay_x_start + overlay_w
                                overlay_y_end = overlay_y_start + overlay_h

                                # Basic boundary check (same as before)
                                if (overlay_y_start >= 0 and overlay_y_end <= frame_h and
                                    overlay_x_start >= 0 and overlay_x_end <= frame_w):

                                    # Copy the *selected* overlay onto the frame ROI
                                    frame[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end] = selected_overlay

                            except Exception as e:
                                print(f"Warning: Error applying conditional overlay for '{icon_key}': {e}")
                    # --- End conditional image overlay ---

                # Update previous scores *after* processing all faces for this frame
                for p_id in persons_seen_this_frame:
                    previous_scores[p_id] = social_score.get(p_id, STARTING_SCORE)

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