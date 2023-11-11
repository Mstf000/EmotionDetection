#Mostafa
import cv2
from deepface import DeepFace

# Load the pre-trained model for emotion detection
emotion_model = DeepFace.build_model('Emotion')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        continue  # Skip the current frame if it's not captured correctly

    # Analyze the frame for emotions
    required_outputs = ['emotion']

    # Set enforce_detection to False to allow the program to continue even if no face is detected
    result = DeepFace.analyze(frame, actions=required_outputs, enforce_detection=False)

    if 'emotion' in result[0]:
        # Get the emotion label if it's detected
        emotion = result[0]['dominant_emotion']

        # Display the frame with emotion detection
        cv2.putText(frame, emotion, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Handle the case when no emotion is detected
        cv2.putText(frame, "No emotion detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Emotion Detection', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
