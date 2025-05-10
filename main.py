import cv2 # OpenCv library handle video input/output and UI

# contain functions for cursor and gesture control
from cursor_movement import handle_cursor_control 
from hand_gesture import handle_gesture_control
import mediapipe as mp # Track hand landmarks

# for array operations and timing logic
import numpy as np 
import time

cap = cv2.VideoCapture(0) # opens the webcam

# intialize mediapipe hand tracking model 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)

# starts with cursor mode
mode = "cursor"  
last_switch_time = 0
switch_cooldown = 1.5  # Cooldown period in seconds to avoid accidental switching - this is the time between modes switching

# For smooth text display
display_text = "Cursor Movement Active"
last_action_time = 0
display_duration = 1.5 


# function to dectect is_thump_up
def is_thumbs_up(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]

    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    # Thumb should be pointing upwards, and other fingers should be curled
    thumb_straight = thumb_tip.y < thumb_ip.y < thumb_mcp.y
    fingers_curled = (index_tip.y > thumb_mcp.y and
                      middle_tip.y > thumb_mcp.y and
                      ring_tip.y > thumb_mcp.y and
                      pinky_tip.y > thumb_mcp.y)
    
    return thumb_straight and fingers_curled

# continuously read frames from the webcam while the camera is opened
while cap.isOpened():
    ret, frame = cap.read()

    # frame is not read properly
    if not ret:
        break
    
    # flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # convert the BGR image to RGB because mediapipe requires RGB images.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_time = time.time()

    # Check for thumb up gesture to switch mode
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # checking that is_thumb_up
            if is_thumbs_up(hand_landmarks.landmark):
                # subtracting current_time from last_switch_time if it is greater then switch_cooldown then mode is 'gesture' if mode is 'cursor'
                if (current_time - last_switch_time) > switch_cooldown:
                    mode = "gesture" if mode == "cursor" else "cursor"
                    last_switch_time = current_time
                    last_action_time = current_time


    # if mode is 'cursor' then call the handle_cursor_control function and pass it necessary argument.
    if mode == "cursor":
        handle_cursor_control(frame, results)
        if (current_time - last_action_time) > display_duration:
            display_text = "Cursor Movement Active"
    # if mode is not 'cursor' then mode is 'gesture' call the handle_gesture_control function and pass it necessary argument.
    elif mode == "gesture":
        gesture_action = handle_gesture_control(frame, results)
        if gesture_action:
            display_text = f"Gesture Detected: {gesture_action}"
            last_action_time = current_time

    cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Hand Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
