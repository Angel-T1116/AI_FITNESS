import os
import streamlit as st
from utils import set_sidebar_visibility 
from utils import get_theme, inject_custom_css

theme = get_theme()
inject_custom_css(theme)


st.title('AI Trainer Demo')


recorded_file = r'C:\Users\angel\OneDrive\Desktop\AI\AI_Final\AI_Personal_Trainer\output_sample.mp4'
if os.path.exists(recorded_file):
    st.video(recorded_file)
else:
    st.error(f"Video file not found: {recorded_file}")

    
    