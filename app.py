from flask import Flask, render_template, request, Response, redirect, url_for, session
import cv2
import mediapipe as mp
import threading
import time
import bicepReps
import squats
import io
from PIL import Image
import math

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a strong, randomly generated key

stop_event = threading.Event()

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    if request.method == "POST":
        exercise = request.form["exercise"]
        duration = int(request.form["duration"])
        reps, accuracy = run_exercise(exercise, duration)
        results = {"exercise": exercise, "reps": reps, "accuracy": accuracy, "duration": duration}
        session.pop('reps', None)
    return render_template("index.html", results=results)

def run_exercise(exercise, duration):
    global stop_event
    stop_event.clear()
    reps = 0
    accuracy = 0

    if exercise == "bicep_curls":
        reps = track_exercise(bicepReps.track_bicep_curls, duration)
    elif exercise == "squats":
        reps = track_exercise(squats.track_squats, duration)

    ideal_reps = math.floor(duration / 3)
    if ideal_reps > 0:
        accuracy = (reps / ideal_reps) * 100
    return reps, accuracy


def track_exercise(tracking_func, duration):
    global stop_event
    reps = 0
    try:
        reps = tracking_func(duration, stop_event)
    except Exception as e:
        print(f"Error during exercise tracking: {e}")
    return reps


def generate_frames(tracking_func, duration):
    global stop_event
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    reps = 0
    try:
        reps = tracking_func(duration, stop_event)
    except Exception as e:
        print(f"Error in exercise tracking: {e}")
        return

    session['reps'] = reps

    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        elapsed_time = time.time() - start_time
        if elapsed_time > duration:
            break

        cv2.putText(frame, f"Time: {int(elapsed_time)}/{duration}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Reps: {reps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        img = Image.fromarray(frame)
        buffer = io.BytesIO()
        img.save(buffer, 'jpeg')
        buffer.seek(0)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.read() + b'\r\n')
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    app.run(debug=True)