import av
import os
import sys
import tempfile
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer
from aiortc.contrib.media import MediaRecorder
from utils import set_sidebar_visibility  
from pathlib import Path
from utils import get_theme, inject_custom_css

theme = get_theme()
inject_custom_css(theme)

# make sure your project root is on sys.path
BASE_DIR = os.path.abspath(os.path.join(__file__, '../../'))
sys.path.append(BASE_DIR)

from utils import get_mediapipe_pose
from process_frame_lunges import ProcessFrame
from threshold_lunges import get_thresholds_beginner


st.title('Lunges AI Trainer')


thresholds = None 


# load thresholds & MP pose model
thresholds = get_thresholds_beginner()
live_processor = ProcessFrame(thresholds=thresholds, flip_frame=True)
pose = get_mediapipe_pose()

# ─── Choice: Live vs Upload ─────────────────────────────────────────────────────
mode = st.radio("Choose input mode:", ["Live webcam", "Upload video"])

if mode == "Upload video":
    # 1) No type filter here so we can handle extension ourselves
    uploaded = st.file_uploader("Upload a video file")

    if uploaded is None:
        st.info("Choose a video file (mp4, mov, avi, mpeg4) to get started.")
    else:
        # 2) Check the extension
        suffix = Path(uploaded.name).suffix.lower()
        allowed = {".mp4", ".mov", ".avi", ".mpeg4"}
        if suffix not in allowed:
            st.error(f"Unsupported format: {suffix}. Please upload one of {', '.join(allowed)}.")
        else:
            # 3) Save upload to temp file
            import tempfile, os, cv2
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.read())
            tmp.flush()
            input_path = tmp.name
        # 2) Prepare output file + writer
        output_path = os.path.join(tempfile.gettempdir(), "processed_lunges.mp4")
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        st.info("Processing video… this may take a minute")
        # 3) Frame-by-frame processing
        frame_idx = 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        progress = st.progress(0)
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            # convert BGR→RGB, process, then back
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            proc_rgb, _ = live_processor.process(frame_rgb, pose)
            proc_bgr = cv2.cvtColor(proc_rgb, cv2.COLOR_RGB2BGR)
            writer.write(proc_bgr)

            frame_idx += 1
            if total:
                progress.progress(min(frame_idx/total, 1.0))

        cap.release()
        writer.release()
        st.success("Done! Here’s your processed video:")
        st.video(output_path)
        with open(output_path, "rb") as vf:
            st.download_button(
                "📥 Download processed video",
                data=vf,
                file_name=f"processed_{uploaded.name}",
                mime="video/mp4"
            )

else:
    # ─── Live webcam streaming path ────────────────────────────────────────────
    output_live = "output_live.flv"
    def video_frame_callback(frame: av.VideoFrame):
        img = frame.to_ndarray(format="rgb24")
        proc, _ = live_processor.process(img, pose)
        return av.VideoFrame.from_ndarray(proc, format="rgb24")

    def recorder_factory() -> MediaRecorder:
        return MediaRecorder(output_live)

    ctx = webrtc_streamer(
        key="lunges-live",
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video":True, "audio":False},
        video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=False),
        out_recorder_factory=recorder_factory
    )

    # download button for live recording
    if os.path.exists(output_live):
        with open(output_live, "rb") as f:
            if st.download_button("📥 Download live recording", f, file_name="live_lunges.flv"):
                os.remove(output_live)
