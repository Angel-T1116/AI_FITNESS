import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

def draw_rounded_rect(img, rect_start, rect_end, corner_width, box_color):

    x1, y1 = rect_start
    x2, y2 = rect_end
    w = corner_width

    # draw filled rectangles
    cv2.rectangle(img, (x1 + w, y1), (x2 - w, y1 + w), box_color, -1)
    cv2.rectangle(img, (x1 + w, y2 - w), (x2 - w, y2), box_color, -1)
    cv2.rectangle(img, (x1, y1 + w), (x1 + w, y2 - w), box_color, -1)
    cv2.rectangle(img, (x2 - w, y1 + w), (x2, y2 - w), box_color, -1)
    cv2.rectangle(img, (x1 + w, y1 + w), (x2 - w, y2 - w), box_color, -1)


    # draw filled ellipses
    cv2.ellipse(img, (x1 + w, y1 + w), (w, w),
                angle = 0, startAngle = -90, endAngle = -180, color = box_color, thickness = -1)

    cv2.ellipse(img, (x2 - w, y1 + w), (w, w),
                angle = 0, startAngle = 0, endAngle = -90, color = box_color, thickness = -1)

    cv2.ellipse(img, (x1 + w, y2 - w), (w, w),
                angle = 0, startAngle = 90, endAngle = 180, color = box_color, thickness = -1)

    cv2.ellipse(img, (x2 - w, y2 - w), (w, w),
                angle = 0, startAngle = 0, endAngle = 90, color = box_color, thickness = -1)

    return img




def draw_dotted_line(frame, lm_coord, start, end, line_color):
    pix_step = 0

    for i in range(start, end+1, 8):
        cv2.circle(frame, (lm_coord[0], i+pix_step), 2, line_color, -1, lineType=cv2.LINE_AA)

    return frame


def draw_text(
    img,
    msg,
    width = 8,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    pos=(0, 0),
    font_scale=1,
    font_thickness=2,
    text_color=(0, 255, 0),
    text_color_bg=(0, 0, 0),
    box_offset=(20, 10),
):

    offset = box_offset
    x, y = pos
    text_size, _ = cv2.getTextSize(msg, font, font_scale, font_thickness)
    text_w, text_h = text_size
    rec_start = tuple(p - o for p, o in zip(pos, offset))
    rec_end = tuple(m + n - o for m, n, o in zip((x + text_w, y + text_h), offset, (25, 0)))
    
    img = draw_rounded_rect(img, rec_start, rec_end, width, text_color_bg)


    cv2.putText(
        img,
        msg,
        (int(rec_start[0] + 6), int(y + text_h + font_scale - 1)), 
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )

    
    return text_size




def find_angle(p1, p2, ref_pt = np.array([0,0])):
    p1_ref = p1 - ref_pt
    p2_ref = p2 - ref_pt

    cos_theta = (np.dot(p1_ref,p2_ref)) / (1.0 * np.linalg.norm(p1_ref) * np.linalg.norm(p2_ref))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            
    degree = int(180 / np.pi) * theta

    return int(degree)





def get_landmark_array(pose_landmark, key, frame_width, frame_height):

    denorm_x = int(pose_landmark[key].x * frame_width)
    denorm_y = int(pose_landmark[key].y * frame_height)

    return np.array([denorm_x, denorm_y])




def get_landmark_features(kp_results, dict_features, feature, frame_width, frame_height):

    if feature == 'nose':
        return get_landmark_array(kp_results, dict_features[feature], frame_width, frame_height)

    elif feature == 'left' or 'right':
        shldr_coord = get_landmark_array(kp_results, dict_features[feature]['shoulder'], frame_width, frame_height)
        elbow_coord   = get_landmark_array(kp_results, dict_features[feature]['elbow'], frame_width, frame_height)
        wrist_coord   = get_landmark_array(kp_results, dict_features[feature]['wrist'], frame_width, frame_height)
        hip_coord   = get_landmark_array(kp_results, dict_features[feature]['hip'], frame_width, frame_height)
        knee_coord   = get_landmark_array(kp_results, dict_features[feature]['knee'], frame_width, frame_height)
        ankle_coord   = get_landmark_array(kp_results, dict_features[feature]['ankle'], frame_width, frame_height)
        foot_coord   = get_landmark_array(kp_results, dict_features[feature]['foot'], frame_width, frame_height)

        return shldr_coord, elbow_coord, wrist_coord, hip_coord, knee_coord, ankle_coord, foot_coord
    
    else:
       raise ValueError("feature needs to be either 'nose', 'left' or 'right")


def get_mediapipe_pose(
                        static_image_mode = False, 
                        model_complexity = 1,
                        smooth_landmarks = True,
                        min_detection_confidence = 0.5,
                        min_tracking_confidence = 0.5

                      ):
    pose = mp.solutions.pose.Pose(
                                    static_image_mode = static_image_mode,
                                    model_complexity = model_complexity,
                                    smooth_landmarks = smooth_landmarks,
                                    min_detection_confidence = min_detection_confidence,
                                    min_tracking_confidence = min_tracking_confidence
                                 )
    return pose

def set_sidebar_visibility(authentication_status):
    if hasattr(st.sidebar, "visible"):
        st.sidebar.visible = authentication_status

def get_theme():
    return st.sidebar.radio("Select Theme", ["Light", "Dark"], key="theme_select")

def inject_custom_css(theme: str):
    if theme == "Light":
        st.markdown("""
            <style>
            html, body, [class*="st-"] {
                color: black !important;
                background-color: #F7F9FC !important;
            }
            .stButton > button {
                background-color: #3A86FF;
                color: white;
                font-weight: bold;
                border: none;
                padding: 0.5em 1.2em;
                border-radius: 8px;
                margin-top: 0.5em;
            }
            .stAlert {
                background-color: #e6f0ff !important;
                color: #003366 !important;
                border-radius: 8px !important;
            }
            section[data-testid="stSidebar"] {
                background-color: #f0f4ff;
                color: black !important;
            }
                    input[type="checkbox"] {
    appearance: none;
    -webkit-appearance: none;
    background-color: #fff;
    border: 2px solid #3A86FF;
    width: 16px;
    height: 16px;
    border-radius: 4px;
    display: inline-block;
    position: relative;
    margin-right: 8px;
    cursor: pointer;
}

/* Blue checkmark when checked */
input[type="checkbox"]:checked::before {
    content: "✔";
    display: block;
    color: #3A86FF;
    font-size: 14px;
    text-align: center;
    line-height: 16px;
    position: absolute;
    top: 0;
    left: 0;
}

            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            html, body, [class*="st-"] {
                color: white !important;
                background-color: #0e0e0e !important;
            }
            .stButton > button {
                background-color: #3A86FF;
                color: white;
                font-weight: bold;
                border: none;
                padding: 0.5em 1.2em;
                border-radius: 8px;
                margin-top: 0.5em;
            }
            .stAlert {
                background-color: #223 !important;
                color: #eee !important;
                border-radius: 8px !important;
            }
            section[data-testid="stSidebar"] {
                background-color: #111;
                color: white !important;
            }
                    input[type="checkbox"] {
    appearance: none;
    -webkit-appearance: none;
    background-color: #fff;
    border: 2px solid #3A86FF;
    width: 16px;
    height: 16px;
    border-radius: 4px;
    display: inline-block;
    position: relative;
    margin-right: 8px;
    cursor: pointer;
}

/* Blue checkmark when checked */
input[type="checkbox"]:checked::before {
    content: "✔";
    display: block;
    color: #3A86FF;
    font-size: 14px;
    text-align: center;
    line-height: 16px;
    position: absolute;
    top: 0;
    left: 0;
}

            </style>
        """, unsafe_allow_html=True)
