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
VIDEO_DURATION = 10  # Duration of each video clip in seconds
SCREEN_WIDTH = 640  # Desired screen width for display

# Create a directory to store the output files
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Initialize MediaPipe face mesh and webcam
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=MAX_NUM_FACES)
cap = cv2.VideoCapture(0)

# Get the default camera resolution
cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
screen_height = int(cam_height * SCREEN_WIDTH / cam_width)

# Initialize variables
face_data = []
face_ids = {}
face_count = 0
start_time = None

# Initialize video writer and audio variables
output_video_file = None
output_audio = pydub.AudioSegment.empty()
output_video_start_time = None

# Resize function
def resize_frame(frame):
    return cv2.resize(frame, (SCREEN_WIDTH, screen_height))

while True:
    # Read frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for display
    frame = resize_frame(frame)

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

    # Record video clips with audio when a person's lips are moving
    if len(face_data) > 0:
        if start_time is None:
            start_time = datetime.datetime.now()

        elapsed_time = datetime.datetime.now() - start_time
        if elapsed_time.total_seconds() >= VIDEO_DURATION:
            if output_video_file is not None:
                output_video_file.release()

            # Save the recorded video clip with audio and corresponding timestamp
            output_file = os.path.join(
                output_dir, f"person_{face_ids[0]}_{start_time.strftime('%Y%m%d_%H%M%S')}.mp4")

            # Write video clip
            output_video_file = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"mp4v"), FRAME_RATE,
                                                (SCREEN_WIDTH, screen_height))
            output_video_start_time = datetime.datetime.now()

            # Save the audio clip with the video
            output_audio_file = output_file
            output_audio.export(output_audio_file, format="mp4")

            # Reset the start time and audio segment
            start_time = None
            output_audio = pydub.AudioSegment.empty()

            # Show a message that the video with audio has been saved
            print(f"Video with audio saved: {output_file}")

    # Write frame to the video writer
    if output_video_file is not None:
        output_video_file.write(frame)

    # Record audio when a person's lips are moving
    if len(face_data) > 0:
        if output_audio.empty():
            output_audio_start_time = output_video_start_time

        # Calculate the elapsed time in milliseconds
        elapsed_time = datetime.datetime.now() - output_audio_start_time
        elapsed_time_ms = int(elapsed_time.total_seconds() * 1000)

        # Append the audio frame to the output audio segment
        output_audio += pydub.AudioSegment(frame.tobytes(), frame_rate=FRAME_RATE, sample_width=frame.dtype.itemsize,
                                           channels=frame.shape[2]).set_frame_position(elapsed_time_ms)

    # Check for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
if output_video_file is not None:
    output_video_file.release()
cap.release()
cv2.destroyAllWindows()

