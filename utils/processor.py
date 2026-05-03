import cv2
import numpy as np
import os
import tempfile
import supervision as sv
from collections import deque, defaultdict
from sklearn.cluster import KMeans
from typing import Dict, Tuple, List

from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration
from roboflow import Roboflow

from .visualizer import PitchVisualizer


class JerseyColorExtractor:
    """Extract jersey colors from player bounding boxes"""
    
    def __init__(self):
        self.team_colors: Dict[int, Tuple[int, int, int]] = {}
        self.player_history: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
        self.stability = 0.85  # Lower = faster color adaptation
        
    def extract_color(self, frame, bbox):
        """Extract dominant color from upper body"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(x2, w), min(y2, h)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            
            # Upper 60% (jersey area)
            upper = crop[:max(1, int(crop.shape[0] * 0.6)), :]
            if upper.size == 0:
                return None
            
            # Convert to RGB and reshape
            pixels = cv2.cvtColor(upper, cv2.COLOR_BGR2RGB).reshape(-1, 3)
            
            # Filter shadows and highlights
            brightness = np.mean(pixels, axis=1)
            mask = (brightness > 30) & (brightness < 225)
            filtered = pixels[mask]
            
            if len(filtered) < 50:
                return None
            
            # KMeans for dominant color
            km = KMeans(n_clusters=2, random_state=42, n_init=5)
            km.fit(filtered)
            labels, counts = np.unique(km.labels_, return_counts=True)
            dominant = km.cluster_centers_[labels[np.argmax(counts)]]
            
            return tuple(map(int, dominant[::-1]))  # RGB to BGR
        except Exception:
            return None
    
    def is_grass(self, color):
        """Check if color is grass"""
        if color is None:
            return True
        b, g, r = color
        return g > r and g > b and g > 80
    
    def update_teams(self, frame, detections, tracker_ids, team_ids):
        """Update team colors from current frame"""
        samples = defaultdict(list)
        
        for bbox, tid, team_id in zip(detections.xyxy, tracker_ids, team_ids):
            color = self.extract_color(frame, bbox)
            if color and not self.is_grass(color):
                samples[int(team_id)].append(color)
                self.player_history[tid].append(color)
                if len(self.player_history[tid]) > 15:
                    self.player_history[tid].pop(0)
        
        for team_id, colors in samples.items():
            if len(colors) >= 3:
                median = tuple(map(int, np.median(colors, axis=0)))
                if team_id in self.team_colors:
                    prev = np.array(self.team_colors[team_id])
                    new = np.array(median)
                    smoothed = (prev * self.stability + new * (1 - self.stability)).astype(int)
                    self.team_colors[team_id] = tuple(smoothed)
                else:
                    self.team_colors[team_id] = median
        
        # Defaults
        if 0 not in self.team_colors:
            self.team_colors[0] = (255, 50, 50)   # Red-ish
        if 1 not in self.team_colors:
            self.team_colors[1] = (50, 50, 255)   # Blue-ish
        
        return self.team_colors


class SportsAnalyticsProcessor:
    """Main video processor with homography-based bird's eye view"""
    
    BALL_ID = 0
    GOALKEEPER_ID = 1
    PLAYER_ID = 2
    REFEREE_ID = 3
    
    def __init__(self, api_key: str):
        self.config = SoccerPitchConfiguration()
        self.colors = JerseyColorExtractor()
        self.viz = PitchVisualizer()
        self.team_classifier = None
        self.tracker = None
        self.homography_buffer = deque(maxlen=5)
        
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
    
    def _predict(self, model, frame, confidence=25):
        """Run Roboflow prediction (lower confidence = faster)"""
        # Save frame temporarily
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
        """Convert Roboflow predictions to Supervision Detections"""
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
    
    def reset(self):
        """Reset tracker and state"""
        self.tracker = sv.ByteTrack()
        self.colors = JerseyColorExtractor()
        self.homography_buffer.clear()
    
    def train_teams(self, video_path, num_frames=15):
        """Quick team classifier training"""
        print("Learning team colors...")
        
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = max(1, total // num_frames)
        
        crops = []
        for i in range(0, total, stride):
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
            
            if len(crops) >= 80:
                break
        cap.release()
        
        if len(crops) >= 8:
            self.team_classifier = TeamClassifier(device="cpu")
            self.team_classifier.fit(crops[:150])
            print(f"Trained with {len(crops)} samples")
            return True
        print(f"Only {len(crops)} samples")
        return False
    
    def compute_homography(self, frame):
        """Compute homography matrix from field keypoints"""
        try:
            preds = self._predict(self.field_model, frame, confidence=20)
            if not preds or len(preds.get('predictions', [])) < 4:
                return None
            
            # Extract keypoints
            cam_pts = []
            confs = []
            for p in preds['predictions']:
                cam_pts.append([p['x'], p['y']])
                confs.append(p['confidence'])
            
            cam_pts = np.array(cam_pts, dtype=np.float32)
            confs = np.array(confs)
            
            # Filter high confidence
            mask = confs > 0.4  # Lower threshold = more points
            cam_pts_filtered = cam_pts[mask]
            
            # Get corresponding pitch points
            pitch_pts = np.array(self.config.vertices[:27], dtype=np.float32)[mask]
            
            if len(cam_pts_filtered) < 4:
                return None
            
            # Compute homography with RANSAC
            H, inlier_mask = cv2.findHomography(
                cam_pts_filtered, pitch_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0
            )
            
            if H is None:
                return None
            
            # Smooth over time
            self.homography_buffer.append(H)
            H_smooth = np.mean(np.array(self.homography_buffer), axis=0)
            H_smooth = H_smooth / H_smooth[2, 2]  # Normalize
            
            return H_smooth
            
        except Exception:
            return None
    
    def apply_homography(self, H, points):
        """Transform points using homography matrix"""
        if len(points) == 0:
            return np.array([])
        
        # Homogeneous coordinates
        ones = np.ones((len(points), 1))
        homogeneous = np.hstack([points, ones])
        
        # Apply transform
        transformed = H @ homogeneous.T
        transformed = transformed / transformed[2, :]
        
        return transformed[:2, :].T
    
    def process_frame(self, frame, frame_count=0):
        """Process single frame"""
        if self.tracker is None:
            self.reset()
        
        # Detect players
        preds = self._predict(self.player_model, frame)
        dets = self._to_detections(preds)
        
        if len(dets) == 0:
            return frame, None
        
        # Split ball from players
        ball = dets[dets.class_id == self.BALL_ID]
        if len(ball) > 0:
            ball.xyxy = sv.pad_boxes(ball.xyxy, px=10)
        
        players = dets[dets.class_id != self.BALL_ID]
        if len(players) == 0:
            return self.viz.annotate_frame(frame, sv.Detections.empty(), 
                                          [], [], {}, ball), None
        
        # Track and classify
        players = players.with_nms(threshold=0.5, class_agnostic=True)
        players = self.tracker.update_with_detections(detections=players)
        
        if len(players) == 0:
            return frame, None
        
        tracker_ids = (players.tracker_id 
                      if hasattr(players, 'tracker_id') 
                      else np.arange(len(players)))
        
        # Team classification
        if self.team_classifier is not None:
            try:
                pmask = players.class_id == self.PLAYER_ID
                if pmask.sum() > 0:
                    pcrops = [sv.crop_image(frame, b) for b in players.xyxy[pmask]]
                    if pcrops:
                        players.class_id[pmask] = self.team_classifier.predict(pcrops)
            except Exception:
                pass
        
        # Update jersey colors
        team_colors = self.colors.update_teams(
            frame, players, tracker_ids, players.class_id
        )
        
        # Annotate frame
        annotated = self.viz.annotate_frame(
            frame, players, players.class_id, 
            tracker_ids, team_colors, ball
        )
        
        # Bird's eye view (only every 3rd frame for speed)
        birdseye = None
        if frame_count % 3 == 0:
            H = self.compute_homography(frame)
            if H is not None:
                player_pos = players.get_anchors_coordinates(
                    sv.Position.BOTTOM_CENTER
                )
                pitch_pos = self.apply_homography(H, player_pos)
                
                pitch_ball = None
                if len(ball) > 0:
                    ball_pos = ball.get_anchors_coordinates(
                        sv.Position.BOTTOM_CENTER
                    )
                    pitch_ball = self.apply_homography(H, ball_pos)
                
                birdseye = self.viz.create_birdseye(
                    pitch_pos, players.class_id, team_colors, pitch_ball
                )
        
        return annotated, birdseye
    
    def process_video(self, video_path, progress_callback=None):
        """Process entire video file"""
        self.reset()
        
        # Train team classifier
        self.train_teams(video_path, num_frames=12)
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Output files
        out_annotated = tempfile.NamedTemporaryFile(
            suffix='_tracked.mp4', delete=False
        )
        out_birdseye = tempfile.NamedTemporaryFile(
            suffix='_birdseye.mp4', delete=False
        )
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer_a = cv2.VideoWriter(out_annotated.name, fourcc, fps, (width, height))
        writer_b = cv2.VideoWriter(
            out_birdseye.name, fourcc, fps,
            (int(self.config.length * 0.1), int(self.config.width * 0.1))
        )
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated, birdseye = self.process_frame(frame, frame_count)
            
            writer_a.write(annotated)
            
            if birdseye is not None:
                writer_b.write(birdseye)
            else:
                writer_b.write(self.viz.create_blank_pitch())
            
            frame_count += 1
            
            if progress_callback and frame_count % 10 == 0:
                progress_callback(frame_count / total_frames)
        
        cap.release()
        writer_a.release()
        writer_b.release()
        
        return out_annotated.name, out_birdseye.name, frame_count
