import cv2
import mediapipe as mp

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
draw_points = []  # Stores ((x, y), is_drawing)

# Helper function to check if a finger is up
def finger_up(hand_landmarks, tip_id, pip_id):
    return hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y

# Count how many fingers are up (thumb, index, middle, ring, pinky)
def count_raised_fingers(hand_landmarks):
    fingers = []

    # Thumb: compare tip and joint in x-axis (flipped frame)
    thumb_is_open = hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x
    fingers.append(thumb_is_open)

    # Other fingers: tip y < pip y
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        fingers.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y)

    return sum(fingers)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            index_tip = hand_landmarks.landmark[8]
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)

            # Count fingers up
            fingers_up = count_raised_fingers(hand_landmarks)
            index_up = finger_up(hand_landmarks, 8, 6)

            # Drawing mode: index finger only
            if index_up and fingers_up < 5:
                draw_points.append(((ix, iy), True))
                cv2.circle(frame, (ix, iy), 5, (0, 0, 255), -1)

            # Erase mode: all 5 fingers up
            elif fingers_up == 5:
                draw_points = [pt for pt in draw_points if abs(pt[0][0] - ix) > 30 or abs(pt[0][1] - iy) > 30]
                cv2.circle(frame, (ix, iy), 30, (255, 255, 255), -1)

            else:
                draw_points.append(((0, 0), False))  # Break stroke

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw lines between points
    for i in range(1, len(draw_points)):
        (pt1, draw1) = draw_points[i - 1]
        (pt2, draw2) = draw_points[i]
        if draw1 and draw2:
            cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

    cv2.imshow("Finger Drawing (Erase on 5 Fingers)", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        draw_points = []

cap.release()
cv2.destroyAllWindows()
