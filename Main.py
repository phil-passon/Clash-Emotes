import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import math

from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import face_mesh as mp_face_mesh
from mediapipe.python.solutions import drawing_utils as mp_draw
from mediapipe.python.solutions import face_mesh_connections as mp_connections

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.7)

img_boohoo = cv2.imread('Assets/boohoo.png')
img_excited = cv2.imread('Assets/excited.png')
img_flex = cv2.imread('Assets/flex.png')
img_hide = cv2.imread('Assets/hide.png')
img_neutral = cv2.imread('Assets/neutral.png')
img_peace = cv2.imread('Assets/peace.png')
img_waiting = cv2.imread('Assets/waiting.png')
img_smile = cv2.imread('Assets/smile.png')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    display_img = frame.copy()
    skeleton_img = frame.copy()

    emote_active = False
    face_center_x, face_center_y = 0.5, 0.5
    mouth_x, mouth_y = 0.5, 0.5
    eye_y = 0.5
    is_smiling = False

    # --- FACE LOGIC ---
    if face_results.multi_face_landmarks:
        for face_lms in face_results.multi_face_landmarks:
            mp_draw.draw_landmarks(skeleton_img, face_lms, mp_connections.FACEMESH_TESSELATION,
                                   None, mp_draw.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=1))

            face_center_x = face_lms.landmark[1].x  # Nose Tip
            face_center_y = face_lms.landmark[1].y
            mouth_x = face_lms.landmark[13].x
            mouth_y = face_lms.landmark[13].y
            eye_y = face_lms.landmark[33].y

            face_left_edge = face_lms.landmark[234].x
            face_right_edge = face_lms.landmark[454].x

            m_left, m_right = face_lms.landmark[61], face_lms.landmark[291]
            e_left, e_right = face_lms.landmark[33], face_lms.landmark[263]
            smile_ratio = abs(m_left.x - m_right.x) / abs(e_left.x - e_right.x)

            if smile_ratio > 0.70:
                is_smiling = True

    # --- HAND LOGIC VARIABLES ---
    fists_near_mouth = 0
    hands_near_nose_for_hide = 0
    fists_in_flex_pos = 0
    peace_sign_detected = False
    hand_on_chin_waiting = False
    hand_near_mouth_excited = False

    # --- HAND LOGIC ---
    if hand_results.multi_hand_landmarks:
        num_hands_detected = len(hand_results.multi_hand_landmarks)

        for hand_lms in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(skeleton_img, hand_lms, mp_hands.HAND_CONNECTIONS)
            lms = hand_lms.landmark

            palm_x, palm_y = lms[0].x, lms[0].y
            knuckle_x, knuckle_y = lms[5].x, lms[5].y
            index_tip_y = lms[8].y

            # PEACE LOGIC
            index_up, middle_up = lms[8].y < lms[6].y, lms[12].y < lms[10].y
            ring_down, pinky_down = lms[16].y > lms[14].y, lms[20].y > lms[18].y
            if index_up and middle_up and ring_down and pinky_down:
                peace_sign_detected = True

            is_fist = lms[8].y > lms[5].y

            # HIDE PRE-CHECK (Looking for hands near center of face)
            dist_to_nose = math.sqrt((palm_x - face_center_x) ** 2 + (palm_y - face_center_y) ** 2)
            if dist_to_nose < 0.25 and not is_fist:
                hands_near_nose_for_hide += 1

            # BOOHOO LOGIC
            dist_to_mouth = math.sqrt((knuckle_x - mouth_x) ** 2 + (knuckle_y - mouth_y) ** 2)
            if is_fist and dist_to_mouth < 0.18:
                fists_near_mouth += 1

            # FLEX LOGIC
            if is_fist and knuckle_y < eye_y:
                if palm_x < face_left_edge or palm_x > face_right_edge:
                    fists_in_flex_pos += 1

            # WAITING LOGIC: Hand near mouth, fingertips reach eye level
            # Only trigger if NOT doing a two-hand gesture
            if dist_to_mouth < 0.18 and index_tip_y < face_center_y and num_hands_detected == 1:
                hand_on_chin_waiting = True

            # EXCITED LOGIC: Hand near mouth, fingertips below nose tip
            if dist_to_mouth < 0.25 and index_tip_y > face_center_y:
                hand_near_mouth_excited = True

    # --- FINAL PRIORITY CHECK ---
    if hands_near_nose_for_hide == 2 and img_hide is not None:
        display_img = cv2.resize(img_hide, (w, h))
        emote_active = True
    elif hand_on_chin_waiting and img_waiting is not None:
        display_img = cv2.resize(img_waiting, (w, h))
        emote_active = True
    elif peace_sign_detected and img_peace is not None:
        display_img = cv2.resize(img_peace, (w, h))
        emote_active = True
    elif fists_in_flex_pos == 2 and img_flex is not None:
        display_img = cv2.resize(img_flex, (w, h))
        emote_active = True
    elif is_smiling and hand_near_mouth_excited:
        if img_excited is not None:
            display_img = cv2.resize(img_excited, (w, h))
            emote_active = True
    elif fists_near_mouth == 2 and img_boohoo is not None:
        display_img = cv2.resize(img_boohoo, (w, h))
        emote_active = True
    elif is_smiling and img_smile is not None:
        display_img = cv2.resize(img_smile, (w, h))
        emote_active = True

    # --- NEUTRAL FACE LOGIC ---
    if not emote_active and img_neutral is not None:
        display_img = cv2.resize(img_neutral, (w, h))

    cv2.imshow("Output: Clash Emotes", display_img)
    cv2.imshow("Input: Skeleton Tracker", skeleton_img)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()