import cv2
import numpy as np
import os
import tempfile
import supervision as sv
from collections import deque, defaultdict
from sklearn.cluster import KMeans
from typing import Dict, Tuple, List, Optional

from sports.common.team import TeamClassifier
from sports.configs.soccer import SoccerPitchConfiguration
from roboflow import Roboflow

from .visualizer import PitchVisualizer


# ============================================
# JERSEY COLOR EXTRACTOR
# (Matches Google Colab Cell 4)
# ============================================

class JerseyColorExtractor:
    """Extract actual jersey colors from detected players"""
    
    def __init__(self):
        self.team_colors: Dict[int, Tuple[int, int, int]] = {}
        self.player_history: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
        self.stability = 0.85
        
    def extract_dominant_color(self, frame, bbox, n_colors=3):
        """Extract dominant jersey color from upper body region - Matches Colab Cell 4"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(x2, w), min(y2, h)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            player_crop = frame[y1:y2, x1:x2]
            if player_crop.size == 0:
                return None
                
            player_h = player_crop.shape[0]
            upper_body = player_crop[:max(1, int(player_h * 0.6)), :]
            
            if upper_body.size == 0:
                return None
            
            upper_body_rgb = cv2.cvtColor(upper_body, cv2.COLOR_BGR2RGB)
            pixels = upper_body_rgb.reshape(-1, 3)
            
            brightness = np.mean(pixels, axis=1)
            mask = (brightness > 30) & (brightness < 225)
            filtered_pixels = pixels[mask]
            
            if len(filtered_pixels) < 50:
                return None
            
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(filtered_pixels)
            
            labels, counts = np.unique(kmeans.labels_, return_counts=True)
            dominant_idx = labels[np.argmax(counts)]
            dominant_color = kmeans.cluster_centers_[dominant_idx]
            
            # Return BGR format as plain Python ints
            r, g, b = map(int, dominant_color)
            return (b, g, r)
            
        except Exception:
            return None
    
    def is_similar_to_grass(self, color, threshold=100):
        """Check if color is likely grass"""
        if color is None:
            return True
        b, g, r = color
        return g > r and g > b and g > threshold
    
    def get_team_colors_from_frame(self, frame, detections, tracker_ids, team_ids):
        """Extract team colors from current frame detections - Matches Colab Cell 4"""
        team_samples = defaultdict(list)
        
        for bbox, tracker_id, team_id in zip(detections.xyxy, tracker_ids, team_ids):
            color = self.extract_dominant_color(frame, bbox)
            
            if color is not None and not self.is_similar_to_grass(color):
                tid = int(team_id)
                team_samples[tid].append(color)
                self.player_history[tracker_id].append(color)
                
                if len(self.player_history[tracker_id]) > 15:
                    self.player_history[tracker_id].pop(0)
        
        for team_id, colors in team_samples.items():
            if len(colors) >= 3:
                median_color = tuple(int(x) for x in np.median(colors, axis=0))
                if team_id in self.team_colors:
                    prev = np.array(self.team_colors[team_id])
                    new = np.array(median_color)
                    smoothed = (prev * self.stability + new * (1 - self.stability)).astype(int)
                    self.team_colors[team_id] = tuple(int(x) for x in smoothed)
                else:
                    self.team_colors[team_id] = median_color
        
        # Default colors if not enough samples
        if 0 not in self.team_colors:
            self.team_colors[0] = (255, 50, 50)
        if 1 not in self.team_colors:
            self.team_colors[1] = (50, 50, 255)
            
        return self.team_colors
    
    def get_player_color(self, tracker_id):
        """Get the current color for a specific player - Matches Colab Cell 4"""
        if tracker_id in self.player_history:
            colors = self.player_history[tracker_id]
            if colors:
                return tuple(int(x) for x in np.median(colors, axis=0))
        return None


# ============================================
# MAIN SPORTS ANALYTICS PROCESSOR
# (Matches Google Colab Cell 6)
# ============================================

class SportsAnalyticsProcessor:
    """Main video processor with homography-based bird's eye view"""
    
    BALL_ID = 0
    GOALKEEPER_ID = 1
    PLAYER_ID = 2
    REFEREE_ID = 3
    
    def __init__(self, api_key: str):
        self.config = SoccerPitchConfiguration()
        self.jersey_extractor = JerseyColorExtractor()
        self.viz = PitchVisualizer()
        self.team_classifier = None
        self.tracker = None
        self.homography_buffer = deque(maxlen=5)
        self.current_homography = None
        
        # Initialize Roboflow models
        self.rf = Roboflow(api_key=api_key)
        self.player_model = (
            self.rf.workspace()
            .project("football-players-detection-3zvbc")
            .version(11).model
        )
        self.field_model = (
            self.rf.workspace()
            .project("football-field-detection-f07vi")
            .version(14).model
        )
        self.models_loaded = True
        print("Models loaded!")
    
    # ============================================
    # ROBLOFLOW API METHODS
    # ============================================
    
    def _predict(self, model, frame, confidence=25):
        """Run Roboflow prediction"""
        tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(tmp.name, frame)
        try:
            result = model.predict(tmp.name, confidence=confidence).json()
            os.unlink(tmp.name)
            return result
        except Exception:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
            return None
    
    def _to_detections(self, predictions):
        """Convert Roboflow predictions to Supervision Detections - Matches Colab Cell 6"""
        if not predictions or 'predictions' not in predictions:
            return sv.Detections.empty()
        
        preds = predictions['predictions']
        if not preds:
            return sv.Detections.empty()
        
        xyxy, class_id, confidence = [], [], []
        for p in preds:
            x, y, w, h = p['x'], p['y'], p['width'], p['height']
            xyxy.append([x - w/2, y - h/2, x + w/2, y + h/2])
            class_id.append(p['class_id'])
            confidence.append(p['confidence'])
        
        return sv.Detections(
            xyxy=np.array(xyxy),
            class_id=np.array(class_id),
            confidence=np.array(confidence)
        )
    
    # ============================================
    # HOMOGRAPHY METHODS (cv2.findHomography)
    # ============================================
    
    def compute_homography(self, frame):
        """Compute homography matrix from field keypoints using cv2.findHomography"""
        try:
            preds = self._predict(self.field_model, frame, confidence=20)
            if not preds or len(preds.get('predictions', [])) < 4:
                return None
            
            # Extract camera-view keypoints
            cam_pts = []
            confs = []
            for p in preds['predictions']:
                cam_pts.append([p['x'], p['y']])
                confs.append(p['confidence'])
            
            cam_pts = np.array(cam_pts, dtype=np.float32)
            confs = np.array(confs)
            
            # Filter high confidence points
            mask = confs > 0.4
            cam_filtered = cam_pts[mask]
            
            # Get corresponding pitch points
            all_pitch = np.array(self.config.vertices[:27], dtype=np.float32)
            pitch_filtered = all_pitch[mask]
            
            if len(cam_filtered) < 4:
                return None
            
            # Compute homography with RANSAC
            H, _ = cv2.findHomography(
                srcPoints=cam_filtered,
                dstPoints=pitch_filtered,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0
            )
            
            if H is None:
                return None
            
            # Smooth over time
            self.homography_buffer.append(H)
            H_smooth = np.mean(np.array(self.homography_buffer), axis=0)
            H_smooth = H_smooth / H_smooth[2, 2]
            
            return H_smooth
            
        except Exception:
            return None
    
    def transform_points(self, H, points):
        """Apply homography matrix to transform points"""
        if H is None or len(points) == 0:
            return np.array([])
        
        ones = np.ones((len(points), 1))
        homogeneous = np.hstack([points, ones])
        transformed = H @ homogeneous.T
        transformed = transformed / transformed[2, :]
        
        return transformed[:2, :].T
    
    # ============================================
    # TRAINING (Matches Colab Cell 6)
    # ============================================
    
    def train_team_classifier(self, video_path, num_frames=30):
        """Train team classifier on video - Matches original Colab code"""
        print("Learning team jersey colors...")
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = max(1, total_frames // num_frames)
        
        crops = []
        for i in range(0, total_frames, stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            
            preds = self._predict(self.player_model, frame)
            dets = self._to_detections(preds)
            if len(dets) == 0:
                continue
            
            players = dets[dets.class_id == self.PLAYER_ID]
            for bbox in players.xyxy:
                crop = sv.crop_image(frame, bbox)
                if crop is not None and crop.size > 0:
                    crops.append(crop)
            
            if len(crops) >= 100:
                break
        
        cap.release()
        
        if len(crops) >= 10:
            self.team_classifier = TeamClassifier(device="cpu")
            self.team_classifier.fit(crops[:200])
            print(f"Team classifier trained with {len(crops)} samples")
            return True
        
        print(f"Not enough training samples ({len(crops)})")
        return False
    
    # ============================================
    # RESET
    # ============================================
    
    def reset(self):
        """Reset tracker and state"""
        self.tracker = sv.ByteTrack()
        self.jersey_extractor = JerseyColorExtractor()
        self.homography_buffer.clear()
        self.current_homography = None
    
    # ============================================
    # FRAME PROCESSING (Matches Colab Cell 6)
    # ============================================
    
    def process_frame(self, frame, frame_count=0):
        """Process a single frame and return annotated frame + bird's eye view"""
        if self.tracker is None:
            self.reset()
        
        # Default results
        annotated = frame.copy()
        birdseye = None
        
        # Player detection
        try:
            preds = self._predict(self.player_model, frame)
            detections = self._to_detections(preds)
        except Exception:
            return annotated, None
        
        if len(detections) == 0:
            return annotated, None
        
        # Separate ball from players
        ball_detections = detections[detections.class_id == self.BALL_ID]
        if len(ball_detections) > 0:
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
        
        # Process non-ball detections
        player_detections = detections[detections.class_id != self.BALL_ID]
        if len(player_detections) == 0:
            return annotated, None
        
        # NMS and tracking
        player_detections = player_detections.with_nms(threshold=0.5, class_agnostic=True)
        player_detections = self.tracker.update_with_detections(detections=player_detections)
        
        if len(player_detections) == 0:
            return annotated, None
        
        # Get tracker IDs
        tracker_ids = (
            player_detections.tracker_id 
            if hasattr(player_detections, 'tracker_id') 
            else np.arange(len(player_detections))
        )
        
        # Team classification
        if self.team_classifier is not None:
            try:
                pmask = player_detections.class_id == self.PLAYER_ID
                if pmask.sum() > 0:
                    pcrops = [sv.crop_image(frame, b) for b in player_detections.xyxy[pmask]]
                    if pcrops:
                        player_detections.class_id[pmask] = self.team_classifier.predict(pcrops)
            except Exception:
                pass
        
        # Extract jersey colors
        team_colors = self.jersey_extractor.get_team_colors_from_frame(
            frame, player_detections, tracker_ids, player_detections.class_id
        )
        
        # Annotate frame with jersey colors
        annotated = self.viz.annotate_frame(
            frame, player_detections, player_detections.class_id,
            tracker_ids, team_colors, ball_detections
        )
        
        # Draw ball on annotated frame
        for bbox in ball_detections.xyxy:
            x1, y1, x2, y2 = bbox.astype(int)
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            cv2.drawMarker(annotated, center, (0, 255, 255), cv2.MARKER_STAR, 15, 2)
        
        # Bird's eye view using homography
        # Compute homography every 3 frames for speed
        if frame_count % 3 == 0 or self.current_homography is None:
            self.current_homography = self.compute_homography(frame)
        
        if self.current_homography is not None:
            try:
                # Transform player positions
                player_positions = player_detections.get_anchors_coordinates(
                    sv.Position.BOTTOM_CENTER
                )
                pitch_positions = self.transform_points(
                    self.current_homography, player_positions
                )
                
                # Transform ball position
                pitch_ball = None
                if len(ball_detections) > 0:
                    ball_pos = ball_detections.get_anchors_coordinates(
                        sv.Position.BOTTOM_CENTER
                    )
                    pitch_ball = self.transform_points(
                        self.current_homography, ball_pos
                    )
                
                birdseye = self.viz.create_birdseye(
                    pitch_positions, player_detections.class_id,
                    team_colors, pitch_ball
                )
            except Exception:
                birdseye = None
        
        return annotated, birdseye
    
    # ============================================
    # VIDEO PROCESSING
    # ============================================
    
    def process_video(self, video_path, progress_callback=None):
        """Process entire video file and return output paths"""
        self.reset()
        
        # Train team classifier
        self.train_team_classifier(video_path, num_frames=30)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video file")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output files
        out_annotated = tempfile.NamedTemporaryFile(suffix='_tracked.mp4', delete=False)
        out_birdseye = tempfile.NamedTemporaryFile(suffix='_birdseye.mp4', delete=False)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer_a = cv2.VideoWriter(out_annotated.name, fourcc, fps, (width, height))
        writer_b = cv2.VideoWriter(
            out_birdseye.name, fourcc, fps,
            (int(self.config.length * 0.1), int(self.config.width * 0.1))
        )
        
        blank_pitch = self.viz.create_blank_pitch()
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated, birdseye = self.process_frame(frame, frame_count)
            
            # Write annotated frame
            if annotated is not None and annotated.size > 0:
                writer_a.write(annotated)
            else:
                writer_a.write(frame)
            
            # Write bird's eye view
            if birdseye is not None and birdseye.size > 0:
                writer_b.write(birdseye)
            else:
                writer_b.write(blank_pitch)
            
            frame_count += 1
            
            # Update progress
            if progress_callback and frame_count % 5 == 0:
                progress_callback(frame_count / total_frames)
        
        cap.release()
        writer_a.release()
        writer_b.release()
        
        print(f"Processed {frame_count} frames")
        
        return out_annotated.name, out_birdseye.name, frame_count
