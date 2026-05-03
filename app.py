import streamlit as st
import cv2
import os
import tempfile
import time
from pathlib import Path
import numpy as np
from PIL import Image
import io

from utils.processor import SportsAnalyticsProcessor

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Football Player Tracker",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/football2--v1.png", width=80)
    st.title("Settings")
    
    # API Key
    api_key = st.text_input(
        "Roboflow API Key",
        type="password",
        help="Get your free key at roboflow.com"
    )
    
    st.divider()
    
    # Processing options
    st.subheader("Options")
    confidence = st.slider("Detection Confidence", 10, 50, 25, 5,
                          help="Lower = faster but less accurate")
    skip_frames = st.slider("Process every N frames", 1, 5, 1,
                           help="Higher = faster but choppier results")
    
    st.divider()
    st.caption("Made with ❤️ | Free to use")

# ============================================
# MAIN PAGE
# ============================================
st.title("Football Player Tracker")
st.markdown("### AI-powered player tracking with jersey color detection")

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

# ============================================
# VIDEO UPLOAD SECTION
# ============================================
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload a football video",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a match video for processing"
    )

with col2:
    st.markdown("""
    ### What it does:
    - Detects all players
    - Identifies jersey colors
    - Tracks each player
    - Creates tactical view
    
    ### Speed tips:
    - Lower confidence = faster
    - Skip frames = faster
    - Shorter videos = faster
    """)

# ============================================
# PROCESS VIDEO
# ============================================
if uploaded_file and api_key:
    # Save uploaded file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_video.write(uploaded_file.read())
    video_path = temp_video.name
    
    # Display original video
    st.subheader("Original Video")
    st.video(video_path)
    
    # Process button
    if st.button("Process Video", type="primary", use_container_width=True):
        if st.session_state.processor is None:
            st.session_state.processor = SportsAnalyticsProcessor(api_key=api_key)
        
        st.session_state.processing = True
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(val):
            progress_bar.progress(val)
            status_text.text(f"Processing... {val*100:.0f}%")
        
        status_text.text("Learning team colors...")
        
        try:
            # Process video
            start_time = time.time()
            annotated_path, birdseye_path, total_frames = (
                st.session_state.processor.process_video(
                    video_path,
                    progress_callback=update_progress
                )
            )
            elapsed = time.time() - start_time
            
            progress_bar.progress(1.0)
            status_text.text(f"Done! Processed {total_frames} frames in {elapsed:.1f}s")
            
            # ============================================
            # DISPLAY RESULTS
            # ============================================
            st.success(f"Processing complete! ({elapsed:.1f} seconds)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Tracked Video")
                st.video(annotated_path)
                
                # Download button
                with open(annotated_path, 'rb') as f:
                    st.download_button(
                        "Download Tracked Video",
                        f.read(),
                        file_name="tracked_video.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
            
            with col2:
                st.subheader("Bird's Eye View")
                st.video(birdseye_path)
                
                # Download button
                with open(birdseye_path, 'rb') as f:
                    st.download_button(
                        "Download Bird's Eye View",
                        f.read(),
                        file_name="birdseye_view.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
            
            # Cleanup
            st.session_state.processing = False
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.processing = False
    
    # Cleanup temp file
    if not st.session_state.processing:
        try:
            os.unlink(video_path)
        except:
            pass

elif not api_key:
    st.info("Enter your Roboflow API key in the sidebar to get started")
    st.markdown("""
    ### How to get a free API key:
    1. Go to [roboflow.com](https://roboflow.com)
    2. Sign up (free)
    3. Go to Settings → API Keys
    4. Copy your Private API Key
    5. Paste it in the sidebar
    """)

# ============================================
# FOOTER
# ============================================
st.divider()
st.caption("""
**How it works:** Upload a football video → AI detects players → 
Identifies jersey colors → Tracks movement → Creates tactical view
""")
