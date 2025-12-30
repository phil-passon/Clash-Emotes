import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import math

from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import face_mesh as mp_face_mesh
from mediapipe.python.solutions import drawing_utils as mp_draw
from mediapipe.python.solutions import face_mesh_connections as mp_connections

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.7)

img_boohoo = cv2.imread('Assets/boohoo.webp')
img_smile = cv2.imread('Assets/smile.webp')
img_neutral = cv2.imread('Assets/neutral.webp')
img_hide = cv2.imread('Assets/hide.webp')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    # display_img will show the Emote pictures
    # skeleton_img will show the user with landmarks
    display_img = frame.copy()
    skeleton_img = frame.copy()

    emote_active = False
    face_center_x, face_center_y = 0.5, 0.5
    mouth_x, mouth_y = 0.5, 0.5

    # --- FACE LOGIC ---
    if face_results.multi_face_landmarks:
        for face_lms in face_results.multi_face_landmarks:
            mp_draw.draw_landmarks(skeleton_img, face_lms, mp_connections.FACEMESH_TESSELATION,
                                   None, mp_draw.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=1))

            face_center_x = face_lms.landmark[1].x
            face_center_y = face_lms.landmark[1].y
            mouth_x = face_lms.landmark[13].x
            mouth_y = face_lms.landmark[13].y

            m_left, m_right = face_lms.landmark[61], face_lms.landmark[291]
            e_left, e_right = face_lms.landmark[33], face_lms.landmark[263]
            smile_ratio = abs(m_left.x - m_right.x) / abs(e_left.x - e_right.x)

            if smile_ratio > 0.75:
                if img_smile is not None:
                    display_img = cv2.resize(img_smile, (w, h))
                    emote_active = True

    # --- HAND LOGIC ---
    if hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) == 2:
        fists_near_mouth = 0
        open_hands_covering_face = 0

        for hand_lms in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(skeleton_img, hand_lms, mp_hands.HAND_CONNECTIONS)

            palm_x, palm_y = hand_lms.landmark[0].x, hand_lms.landmark[0].y
            knuckle_x, knuckle_y = hand_lms.landmark[5].x, hand_lms.landmark[5].y

            is_fist = hand_lms.landmark[8].y > hand_lms.landmark[5].y

            dist_to_face = math.sqrt((palm_x - face_center_x) ** 2 + (palm_y - face_center_y) ** 2)
            if dist_to_face < 0.2 and not is_fist:
                open_hands_covering_face += 1

            dist_to_mouth = math.sqrt((knuckle_x - mouth_x) ** 2 + (knuckle_y - mouth_y) ** 2)
            if is_fist and dist_to_mouth < 0.15:
                fists_near_mouth += 1

        if open_hands_covering_face == 2 and img_hide is not None:
            display_img = cv2.resize(img_hide, (w, h))
            emote_active = True
        elif fists_near_mouth == 2 and img_boohoo is not None:
            display_img = cv2.resize(img_boohoo, (w, h))
            emote_active = True

    # --- NEUTRAL FACE LOGIC ---
    if not emote_active and img_neutral is not None:
        display_img = cv2.resize(img_neutral, (w, h))

    # --- OUTPUT ---
    cv2.imshow("Output: Clash Emotes", display_img)
    cv2.imshow("Input: Skeleton Tracker", skeleton_img)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()