import cv2
import mediapipe as mp
import numpy as np
import time

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def track_bicep_curls(duration, stop_event):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return 0

    counter = 0 
    stage = "down"
    start_time = time.time()
    cnt = 0  # Counter for bad posture
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and not stop_event.is_set():
            elapsed_time = time.time() - start_time
            if elapsed_time > duration:
                break
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.flip(image, 1)
            
            try:
                landmarks = results.pose_landmarks.landmark
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, 
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                
                # Improved Rep Counting Logic:
                avg_angle = (left_angle + right_angle) / 2

                if avg_angle < 60 and stage == "down":  # Going up
                    stage = "up"
                elif avg_angle > 160 and stage == "up":  # Going down
                    stage = "down"
                    counter += 1

                # Check for improper posture (simplified)
                if 60 < avg_angle < 160:
                    cnt += 1
                else:
                    cnt = 0

            except:
                pass
            
            cv2.putText(frame, f'Reps: {counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Bicep Curl Counter', frame)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return counter