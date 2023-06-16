import mediapipe as mp
import cv2
import numpy as np
import os
import datetime
import pydub
import pydub.playback

# MediaPipe initialization
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Constants
MAX_NUM_FACES = 4  # Maximum number of faces to detect
FRAME_RATE = 30  # Video frame rate
AUDIO_DURATION = 10  # Duration of each audio clip in seconds

# Create a directory to store the output files
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Initialize MediaPipe face mesh and webcam
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=MAX_NUM_FACES)
cap = cv2.VideoCapture(0)

# Initialize variables
face_data = []
face_ids = {}
face_count = 0
start_time = None

while True:
    # Read frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB and process it with MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Clear previous face data
    face_data.clear()

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract face mesh coordinates
            face_points = []
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                face_points.append((x, y))
            face_data.append(face_points)

    # Visualize the face mesh and assign IDs
    annotated_frame = frame.copy()
    for i, face_points in enumerate(face_data):
        # Assign unique IDs to each face
        if i not in face_ids:
            face_ids[i] = face_count
            face_count += 1

        # Draw face landmarks
        for landmark in face_points:
            cv2.circle(annotated_frame, landmark, 1, (0, 255, 0), -1)

    # Display the annotated frame
    cv2.imshow('Face Mesh', annotated_frame)

    # Record audio when a person's lips are moving
    if len(face_data) > 0:
        if start_time is None:
            start_time = datetime.datetime.now()

        elapsed_time = datetime.datetime.now() - start_time
        if elapsed_time.total_seconds() >= AUDIO_DURATION:
            # Save the recorded audio and corresponding video clip
            output_file = os.path.join(
                output_dir, f"person_{face_ids[0]}_{start_time.strftime('%Y%m%d_%H%M%S')}.mp4")
            output_audio_file = output_file.replace(".mp4", ".mp3")

            # Write video clip
            out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"mp4v"), FRAME_RATE, (frame.shape[1], frame.shape[0]))
            out.write(frame)
            out.release()

            # Extract audio clip
            audio = pydub.AudioSegment.silent(duration=AUDIO_DURATION * 1000)
            audio.export(output_audio_file, format="mp3")

            # Play the audio clip
            pydub.playback.play(audio)

            # Reset variables
            start_time = None

    # Check for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

