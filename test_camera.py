import cv2

cap = cv2.VideoCapture(0) # Use 0 for default camera

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

print("Camera opened. Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    try:
        cv2.imshow('Camera Test', frame)
    except Exception as e:
        print(f"Error during imshow: {e}")
        break # Exit if imshow fails

    # Check for 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
print("Releasing camera...")
cap.release()
cv2.destroyAllWindows()
print("Done.") 