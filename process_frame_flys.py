import time
import cv2
import numpy as np
from utils import find_angle, get_landmark_features, draw_text, draw_dotted_line

class ProcessFrame:
    def __init__(self, thresholds, flip_frame=False):
        self.thresholds = thresholds
        self.flip_frame = flip_frame

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.linetype = cv2.LINE_AA
        self.COLORS = {
            'blue': (0, 127, 255),
            'red': (255, 50, 50),
            'green': (0, 255, 127),
            'light_green': (100, 233, 127),
            'yellow': (255, 255, 0),
            'magenta': (255, 0, 255),
            'white': (255,255,255),
            'cyan': (0, 255, 255),
            'light_blue': (102, 204, 255),
            'rose': (255,102,204),
            'aqua': (192, 220, 205)
        }

        self.dict_features = {}
        self.left_features = {
                                'shoulder': 11,
                                'elbow'   : 13,
                                'wrist'   : 15,                    
                                'hip'     : 23,
                                'knee'    : 25,
                                'ankle'   : 27,
                                'foot'    : 31
                             }

        self.right_features = {
                                'shoulder': 12,
                                'elbow'   : 14,
                                'wrist'   : 16,
                                'hip'     : 24,
                                'knee'    : 26,
                                'ankle'   : 28,
                                'foot'    : 32
                              }

        self.dict_features['left'] = self.left_features
        self.dict_features['right'] = self.right_features
        self.dict_features['nose'] = 0

        self.state_tracker = {
            'state_seq': [],
            'start_inactive_time': time.perf_counter(),
            'INACTIVE_TIME': 0.0,
            'prev_state': None,
            'curr_state': None,
            'REP_COUNT': 0
        }

    def _get_state(self, angle):
        if self.thresholds['SHOULDER_ELBOW_ANGLE']['CLOSE'][0] >= angle >= self.thresholds['SHOULDER_ELBOW_ANGLE']['CLOSE'][1]:
            return 'closed'
        elif self.thresholds['SHOULDER_ELBOW_ANGLE']['OPEN'][0] >= angle >= self.thresholds['SHOULDER_ELBOW_ANGLE']['OPEN'][1]:
            return 'open'
        return None

    def _update_state_sequence(self, state):
        if state == 'open' and len(self.state_tracker['state_seq']) == 0:
            self.state_tracker['state_seq'].append(state)
        elif state == 'closed' and len(self.state_tracker['state_seq']) == 1:
            self.state_tracker['state_seq'].append(state)
        elif state == 'open' and len(self.state_tracker['state_seq']) == 2:
            self.state_tracker['REP_COUNT'] += 1
            self.state_tracker['state_seq'] = []

    def process(self, frame, pose):
        frame_height, frame_width, _ = frame.shape
        keypoints = pose.process(frame)
        play_sound = None

        if keypoints.pose_landmarks:
            landmarks = keypoints.pose_landmarks.landmark

            left_shldr, left_elbow, left_wrist = get_landmark_features(landmarks, self.dict_features, 'left', frame_width, frame_height)
            right_shldr, right_elbow, right_wrist = get_landmark_features(landmarks, self.dict_features, 'right', frame_width, frame_height)

            offset_angle = find_angle(left_shldr, right_shldr, get_landmark_features(landmarks, self.dict_features, 'nose', frame_width, frame_height))
            if offset_angle > self.thresholds['OFFSET_THRESH']:
                draw_text(frame, 'CAMERA NOT ALIGNED PROPERLY!!!', pos=(30, frame_height-60), text_color=(255,255,230), font_scale=0.65, text_color_bg=(255, 153, 0))
                draw_text(frame, f'OFFSET ANGLE: {int(offset_angle)}', pos=(30, frame_height-30), text_color=(255,255,230), font_scale=0.65, text_color_bg=(255, 153, 0))
                self.state_tracker['INACTIVE_TIME'] = 0.0
                self.state_tracker['start_inactive_time'] = time.perf_counter()
                return frame, None

            angle_left = find_angle(left_shldr, left_elbow, left_wrist)
            angle_right = find_angle(right_shldr, right_elbow, right_wrist)
            avg_angle = (angle_left + angle_right) / 2

            current_state = self._get_state(avg_angle)
            self.state_tracker['curr_state'] = current_state
            self._update_state_sequence(current_state)

            # Show counters
            draw_text(frame, f"CORRECT: {self.state_tracker['REP_COUNT']}", pos=(int(frame_width*0.68), 30), text_color=(255, 255, 230), font_scale=0.7, text_color_bg=(18, 185, 0))

            # Show angle for debug
            draw_text(frame, f"ANGLE: {int(avg_angle)}", pos=(int(frame_width*0.68), 80), text_color=(255, 255, 230), font_scale=0.7, text_color_bg=(0, 90, 255))

            # Handle inactivity
            if self.state_tracker['curr_state'] == self.state_tracker['prev_state']:
                end_time = time.perf_counter()
                self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']
                self.state_tracker['start_inactive_time'] = end_time
                if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
                    self.state_tracker['REP_COUNT'] = 0
                    draw_text(frame, 'RESETTING COUNT DUE TO INACTIVITY', pos=(30, frame_height-90), text_color=(0,0,0), font_scale=0.6, text_color_bg=(255, 255, 0))
                    play_sound = 'reset_counters'
                    self.state_tracker['INACTIVE_TIME'] = 0.0
            else:
                self.state_tracker['start_inactive_time'] = time.perf_counter()
                self.state_tracker['INACTIVE_TIME'] = 0.0

            self.state_tracker['prev_state'] = current_state
        else:
            draw_text(frame, 'POSE NOT DETECTED', pos=(30, 30), text_color=(255, 255, 230), font_scale=0.65, text_color_bg=(0, 0, 255))

        if self.flip_frame:
            frame = cv2.flip(frame, 1)

        return frame, play_sound
