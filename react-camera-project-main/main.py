
import time
import threading
import cv2
import mediapipe as mp
import pygame
pygame.mixer.init()
SOUNDS = {
    "Pointing_Up": "sounds/d.wav",
    "Victory": "sounds/f.wav",
    "Thumb_Up": "sounds/c.wav",
    "Open_Palm": "sounds/b.wav",
    "Thumb_Down": "sounds/e.wav",
    "Closed_Fist": "sounds/a.wav",
}
def play_sound(sound_file):
    pygame.mixer.music.stop()
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()

MODEL_PATH = "gesture_recognizer.task"

REACTIONS = {
    "Pointing_Up": "reactions/point-monkey.jpeg",
    "Victory": "reactions/dap.jpeg",
    "Thumb_Up": "reactions/heart-meme.jpeg",
    "Open_Palm": "reactions/panic.jpeg",
    "Thumb_Down": "reactions/dizzy.jpeg",
    "Closed_Fist": "reactions/meme-monkey-thinking.jpeg",
}

def show_reaction(frame, reaction, x=20, y=20, width=250):
    h, w = reaction.shape[:2]

    ratio = width / w
    height = int(h * ratio)

    reaction = cv2.resize(reaction, (width, height))

    frame_h, frame_w = frame.shape[:2]

    width = min(width, frame_w)
    height = min(height, frame_h)

    x = max(0, min(x, frame_w - width))
    y = max(0, min(y, frame_h - height))

    reaction = cv2.resize(reaction, (width, height))
    frame[y:y + height, x:x + width] = reaction

    return frame


BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
RunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("الكاميرا ما اشتغلت")

start_time = time.time()
last_gesture = ""
with GestureRecognizer.create_from_options(options) as recognizer:
    while True:
        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp = int((time.time() - start_time) * 1000)

        result = recognizer.recognize_for_video(
            mp_image,
            timestamp
        )

        gesture_name = "No gesture"
        confidence = 0.0

        if result.gestures and result.gestures[0]:
            gesture = result.gestures[0][0]
            gesture_name = gesture.category_name
            confidence = gesture.score

        if gesture_name in REACTIONS:
            reaction = cv2.imread(REACTIONS[gesture_name])

            if reaction is not None:
                frame = show_reaction(frame, reaction)
            if gesture_name != last_gesture:

             play_sound(SOUNDS[gesture_name])

             last_gesture = gesture_name

        else:

         last_gesture = ""

        cv2.putText(
            frame,
            f"{gesture_name} {confidence:.0%}",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        print(gesture_name)

        cv2.imshow("Reaction Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()