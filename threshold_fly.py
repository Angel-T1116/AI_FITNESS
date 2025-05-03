def get_thresholds_beginner():
    """
    Thresholds for dumbbell fly exercise detection.
    Detects opening and closing of arms on the horizontal plane.
    """
    _ANGLE_SHOULDER_ELBOW = {
        'OPEN': (160, 110),   # Arms extended outward
        'CLOSE': (109, 40)    # Arms brought inward toward center
    }

    thresholds = {
        'SHOULDER_ELBOW_ANGLE': _ANGLE_SHOULDER_ELBOW,
        'OFFSET_THRESH': 35.0,         # Pose offset tolerance
        'INACTIVE_THRESH': 15.0,      # Seconds before resetting
        'CNT_FRAME_THRESH': 50        # Frames before removing a prompt
    }

    return thresholds
