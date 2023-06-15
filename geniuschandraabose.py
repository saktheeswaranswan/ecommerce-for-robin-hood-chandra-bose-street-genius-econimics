import cv2
import mediapipe as mp
import numpy as np
import os
import sounddevice as sd
import scipy.io.wavfile as wav

# Initialize MediaPipe FaceMesh
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Constants
NUM_FACE_IDS = 2  # Maximum number of face IDs to track
AUDIO_DURATION = 3  # Audio recording duration in seconds
OUTPUT_FOLDER = "recordings"  # Folder to save the audio recordings

# Create the output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Initialize face mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=NUM_FACE_IDS)

# Function to record audio for a given person ID
def record_audio(person_id):
    print(f"Recording audio for person {person_id}...")
    audio = sd.rec(int(AUDIO_DURATION * 44100), samplerate=44100, channels=1)
    sd.wait()

    # Save the audio recording
    output_path = os.path.join(OUTPUT_FOLDER, f"person_{person_id}.wav")
    wav.write(output_path, 44100, audio)

    print(f"Audio recording for person {person_id} saved at {output_path}")

# Function to check if lip movement is detected for a given face mesh
def has_lip_movement(face_landmarks):
    # Extract lip landmarks
    upper_lip_landmarks = np.array([[landmark.x, landmark.y] for landmark in face_landmarks[61:65]])
    lower_lip_landmarks = np.array([[landmark.x, landmark.y] for landmark in face_landmarks[65:69]])

    # Calculate lip distance
    lip_distance = np.mean(np.linalg.norm(upper_lip_landmarks - lower_lip_landmarks, axis=1))

    return lip_distance > 0.02  # Adjust this threshold as needed

# Create video capture from webcam
cap = cv2.VideoCapture("cmmunist.mp4")

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe FaceMesh
    results = face_mesh.process(image_rgb)

    # Check if any faces were detected
    if results.multi_face_landmarks:
        num_faces = len(results.multi_face_landmarks)
        print(f"Number of faces detected: {num_faces}")

        # Iterate over each detected face
        for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
            # Draw the face landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            )

            # Check if lip movement is detected for the current face
            if has_lip_movement(face_landmarks.landmark):
                # Play audio
                sd.play(sd.rec(int(AUDIO_DURATION * 44100), samplerate=44100, channels=1))
                # Record audio for the corresponding person ID
                record_audio(face_id + 1)
                # Stop playing audio
                sd.stop()

    # Display the frame with face mesh detection
    cv2.imshow("Face Mesh", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

