import os
import streamlit as st
from utils import set_sidebar_visibility  

st.title('AI Fitness Trainer: Squats Analysis')


recorded_file = r'C:\Users\angel\OneDrive\Desktop\AI\AI_Final\AI_Personal_Trainer\output_sample.mp4'
if os.path.exists(recorded_file):
    st.video(recorded_file)
else:
    st.error(f"Video file not found: {recorded_file}")

    
    