import cv2
import mediapipe as mp
import numpy as np
import time

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_angle(A, B, C):
    AB = np.array([B[0] - A[0], B[1] - A[1]])
    BC = np.array([C[0] - B[0], C[1] - B[1]])
    
    dot_product = np.dot(AB, BC)
    magnitude_AB = np.linalg.norm(AB)
    magnitude_BC = np.linalg.norm(BC)

    if magnitude_AB == 0 or magnitude_BC == 0:
        return 0  # Prevent division by zero

    angle_rad = np.arccos(dot_product / (magnitude_AB * magnitude_BC))
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def track_squats(duration, stop_event):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return 0
        
    start_time = None
    workout_started = False
    rep_count = 0
    going_down = False
    workout_timer = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break
                
            h, w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                try:
                    landmarks = results.pose_landmarks.landmark
                    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
                    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
                    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                    nose = landmarks[mp_pose.PoseLandmark.NOSE]
                    
                    left_knee_pixel = (int(left_knee.x * w), int(left_knee.y * h))
                    right_knee_pixel = (int(right_knee.x * w), int(right_knee.y * h))
                    left_hip_pixel = (int(left_hip.x * w), int(left_hip.y * h))
                    right_hip_pixel = (int(right_hip.x * w), int(right_hip.y * h))
                    nose_pixel = (int(nose.x * w), int(nose.y * h))
                    
                    knee_angle = calculate_angle(left_knee_pixel, nose_pixel, right_knee_pixel)
                    left_hip_to_knee = calculate_distance(left_hip_pixel, left_knee_pixel)
                    right_hip_to_knee = calculate_distance(right_hip_pixel, right_knee_pixel)
                    
                    #Check for starting position
                    if 145 <= knee_angle <= 155: #Adjust threshold as needed
                        if start_time is None:
                            start_time = time.time()
                        elif time.time() - start_time >= 1 and not workout_started:
                            workout_started = True
                            workout_timer = time.time()  # Start workout timer
                    else:
                        start_time = None
                    
                    if workout_started:
                        elapsed_time = int(time.time() - workout_timer)
                        remaining_time = max(0, duration - elapsed_time)
                        cv2.putText(frame, f'Time Left: {remaining_time}s', (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        
                        if left_hip_to_knee <= 116 or right_hip_to_knee <= 116:
                            going_down = True
                        
                        if going_down and (left_hip_to_knee >= 125 and right_hip_to_knee >= 125):
                            rep_count += 1
                            going_down = False  # Reset for the next rep
                            time.sleep(0.3)  # Prevent accidental multiple counts
                        
                        if remaining_time == 0:
                            break  # End workout session
                        
                        cv2.putText(frame, f'Reps: {rep_count}', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.putText(frame, f'Knee Angle: {knee_angle:.2f}°', (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        cv2.putText(frame, f'L Hip-Knee: {left_hip_to_knee:.2f}px', (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.putText(frame, f'R Hip-Knee: {right_hip_to_knee:.2f}px', (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                except Exception as e:
                    print(f"Error processing landmarks: {e}")
                    pass
            
            cv2.imshow('Squat Counter', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return rep_count