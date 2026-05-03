import streamlit as st
import cv2
import os
import tempfile
import time
import numpy as np
from utils.processor import SportsAnalyticsProcessor

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="⚽ Football Player Tracker",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# GET API KEYS FROM SECRETS
# ============================================
def get_secrets():
    """Get API keys from Streamlit secrets or environment variables"""
    api_key = None
    hf_token = None
    
    # Try Streamlit Cloud secrets first
    try:
        api_key = st.secrets["ROBOFLOW_API_KEY"]
    except:
        pass
    
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except:
        pass
    
    # Fallback to environment variables (for local testing)
    if not api_key:
        api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN")
    
    # Set HF token for transformers/sports library
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    
    return api_key, hf_token

API_KEY, HF_TOKEN = get_secrets()

# ============================================
# INITIALIZE SESSION STATE
# ============================================
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'results' not in st.session_state:
    st.session_state.results = None

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/football2--v1.png", width=80)
    st.title("Settings")
    
    # API Status
    st.subheader("API Status")
    if API_KEY:
        st.success("Roboflow API ready")
    else:
        st.error("Missing ROBOFLOW_API_KEY")
    
    if HF_TOKEN:
        st.success("Hugging Face Token ready")
    else:
        st.info("HF Token optional (faster downloads)")
    
    st.divider()
    
    # Processing options
    st.subheader("Performance")
    confidence = st.slider(
        "Detection Confidence",
        min_value=10,
        max_value=50,
        value=25,
        step=5,
        help="Lower = faster but less accurate"
    )
    
    skip_frames = st.slider(
        "Process Every N Frames",
        min_value=1,
        max_value=5,
        value=1,
        help="Higher = faster but choppier output"
    )
    
    st.divider()
    
    st.caption("""
    **Best Results:**
    - Videos under 5 min
    - Clear pitch view
    - Good lighting
    - Steady camera
    """)
    
    st.divider()
    st.caption("Made with ❤️ | Free")

# ============================================
# MAIN PAGE
# ============================================
st.title("⚽ Football Player Tracker")
st.markdown("""
### AI-Powered Player Tracking with Jersey Color Detection

Upload a match video and get real jersey color tracking + tactical bird's eye view.
""")

# ============================================
# VIDEO UPLOAD SECTION
# ============================================
if API_KEY:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Football Video",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            help="Upload a match video (max 5 min recommended)"
        )
    
    with col2:
        st.info("""
        **Recommended:**
        - 720p or 1080p
        - 1-5 minutes long
        - MP4 format
        - Clear pitch view
        
        Need samples? Try [YouTube](https://youtube.com)
        """)
    
    # ============================================
    # PROCESS UPLOADED VIDEO
    # ============================================
    if uploaded_file:
        # Save temporarily
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frames / fps if fps > 0 else 0
            cap.release()
            
            # Display info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Duration", f"{duration:.1f}s")
            with col2:
                st.metric("Frames", f"{frames:,}")
            with col3:
                st.metric("Resolution", f"{width}x{height}")
            with col4:
                st.metric("FPS", f"{fps:.0f}")
        
        st.divider()
        
        # Process button
        process_col1, process_col2 = st.columns([2, 1])
        
        with process_col1:
            process_btn = st.button(
                "Process Video",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.processing
            )
        
        with process_col2:
            if st.session_state.processing:
                st.info("Processing...")
        
        # ============================================
        # VIDEO PROCESSING
        # ============================================
        if process_btn:
            if st.session_state.processor is None:
                st.session_state.processor = SportsAnalyticsProcessor(
                    api_key=API_KEY
                )
            
            st.session_state.processing = True
            st.session_state.results = None
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_text = st.empty()
            
            def update_progress(val):
                progress_bar.progress(min(val, 1.0))
                status_text.text(f"Processing... {val*100:.0f}%")
            
            try:
                start_time = time.time()
                status_text.text("Learning team jersey colors...")
                
                annotated_path, birdseye_path, total_frames = (
                    st.session_state.processor.process_video(
                        video_path,
                        progress_callback=update_progress
                    )
                )
                
                elapsed = time.time() - start_time
                
                st.session_state.results = {
                    'annotated': annotated_path,
                    'birdseye': birdseye_path,
                    'frames': total_frames,
                    'time': elapsed
                }
                
                progress_bar.progress(1.0)
                status_text.text("Complete!")
                time_text.text(f"{total_frames} frames in {elapsed:.1f}s")
                
                st.session_state.processing = False
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.processing = False
        
        # ============================================
        # DISPLAY RESULTS
        # ============================================
        if st.session_state.results:
            results = st.session_state.results
            
            st.success(f"Done in {results['time']:.1f}s")
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Tracked Video")
                with open(results['annotated'], 'rb') as f:
                    video_bytes = f.read()
                st.video(video_bytes)
                st.download_button(
                    "Download Tracked Video",
                    video_bytes,
                    "tracked_video.mp4",
                    "video/mp4",
                    use_container_width=True
                )
            
            with col2:
                st.subheader("Bird's Eye View")
                with open(results['birdseye'], 'rb') as f:
                    birdseye_bytes = f.read()
                st.video(birdseye_bytes)
                st.download_button(
                    "Download Bird's Eye View",
                    birdseye_bytes,
                    "birdseye_view.mp4",
                    "video/mp4",
                    use_container_width=True
                )
            
            # Stats
            st.divider()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Frames", f"{results['frames']:,}")
            with col2:
                st.metric("Time", f"{results['time']:.1f}s")
            with col3:
                fps = results['frames'] / results['time'] if results['time'] > 0 else 0
                st.metric("Speed", f"{fps:.1f} FPS")
            
            if st.button("Process Another", use_container_width=True):
                st.session_state.results = None
                st.rerun()
        
        # Cleanup
        if not st.session_state.processing:
            try:
                os.unlink(video_path)
            except:
                pass

else:
    # ============================================
    # NO API KEY - SHOW SETUP INSTRUCTIONS
    # ============================================
    st.warning("API key not configured!")
