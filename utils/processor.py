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


class JerseyColorExtractor:
    """Extract actual jersey colors from detected players"""
    
    def __init__(self):
        self.team_colors: Dict[int, Tuple[int, int, int]] = {}
        self.player_history: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
        self.stability = 0.85
        
    def extract_dominant_color(self, frame, bbox, n_colors=3):
        """Extract dominant jersey color from upper body region"""
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
            
            # Return BGR as Python ints
            r_val, g_val, b_val = map(int, dominant_color)
            return (b_val, g_val, r_val)
            
        except Exception:
            return None
    
    def is_similar_to_grass(self, color, threshold=100):
        """Check if color is likely grass"""
        if color is None:
            return True
        b_val, g_val, r_val = color
        return g_val > r_val and g_val > b_val and g_val > threshold
    
    def get_team_colors_from_frame(self, frame, detections, tracker_ids, team_ids):
        """Extract team colors from current frame"""
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
                    prev = np.array(self.team_colors[team_id], dtype=np.float64)
                    new = np.array(median_color, dtype=np.float64)
                    smoothed = prev * self.stability + new * (1.0 - self.stability)
                    self.team_colors[team_id] = (int(smoothed[0]), int(smoothed[1]), int(smoothed[2]))
                else:
                    self.team_colors[team_id] = median_color
        
        # Default colors
        if 0 not in self.team_colors:
            self.team_colors[0] = (255, 50, 50)
        if 1 not in self.team_colors:
            self.team_colors[1] = (50, 50, 255)
            
        return self.team_colors


class SportsAnalyticsProcessor:
    """Main video processor with homography-based bird's eye view"""
    
    BALL_ID = 0
    PLAYER_ID = 2
    
    def __init__(self, api_key: str):
        self.config = SoccerPitchConfiguration()
        self.colors = JerseyColorExtractor()
        self.viz = PitchVisualizer()
        self.team_classifier = None
        self.tracker = None
        self.homography_buffer = deque(maxlen=5)
        self.current_H = None
        
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
            xyxy=np.array(xyxy, dtype=np.float32),
            class_id=np.array(class_id, dtype=np.int32),
            confidence=np.array(confidence, dtype=np.float32)
        )
    
    def compute_homography(self, frame):
        """Compute homography matrix using cv2.findHomography"""
        try:
            preds = self._predict(self.field_model, frame, confidence=20)
            if not preds or len(preds.get('predictions', [])) < 4:
                return None
            
            cam_pts = []
            confs = []
            for p in preds['predictions']:
                cam_pts.append([p['x'], p['y']])
                confs.append(p['confidence'])
            
            cam_pts = np.array(cam_pts, dtype=np.float32)
            confs = np.array(confs, dtype=np.float32)
            
            mask = confs > 0.4
            if mask.sum() < 4:
                return None
            
            cam_filtered = cam_pts[mask]
            pitch_filtered = np.array(self.config.vertices[:27], dtype=np.float32)[mask]
            
            H, _ = cv2.findHomography(
                srcPoints=cam_filtered,
                dstPoints=pitch_filtered,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0
            )
            
            if H is None:
                return None
            
            self.homography_buffer.append(H)
            H_smooth = np.mean(np.array(self.homography_buffer), axis=0)
            H_smooth = H_smooth / H_smooth[2, 2]
            
            return H_smooth.astype(np.float32)
            
        except Exception:
            return None
    
    def transform_points(self, H, points):
        """Apply homography to transform points"""
        if H is None or len(points) == 0:
            return np.array([])
        
        ones = np.ones((len(points), 1), dtype=np.float32)
        homogeneous = np.hstack([points, ones])
        transformed = H @ homogeneous.T
        transformed = transformed / transformed[2, :]
        
        return transformed[:2, :].T
    
    def reset(self):
        """Reset tracker and state"""
        self.tracker = sv.ByteTrack()
        self.colors = JerseyColorExtractor()
        self.homography_buffer.clear()
        self.current_H = None
    
    def train_teams(self, video_path, num_frames=30):
        """Train team classifier on video"""
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
            print(f"Trained with {len(crops)} samples")
            return True
        
        print(f"Only {len(crops)} samples")
        return False
    
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
            return frame, None
        
        # Track and classify
        players = players.with_nms(threshold=0.5, class_agnostic=True)
        players = self.tracker.update_with_detections(detections=players)
        
        if len(players) == 0:
            return frame, None
        
        tracker_ids = (
            players.tracker_id 
            if hasattr(players, 'tracker_id') 
            else np.arange(len(players))
        )
        
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
        
        # Jersey colors
        team_colors = self.colors.get_team_colors_from_frame(
            frame, players, tracker_ids, players.class_id
        )
        
        # Annotate frame
        annotated = self.viz.annotate_frame(
            frame, players, players.class_id,
            tracker_ids, team_colors, ball
        )
        
        # Bird's eye view (every 3 frames)
        birdseye = None
        if frame_count % 3 == 0 or self.current_H is None:
            self.current_H = self.compute_homography(frame)
        
        if self.current_H is not None:
            try:
                player_pos = players.get_anchors_coordinates(
                    sv.Position.BOTTOM_CENTER
                )
                pitch_pos = self.transform_points(self.current_H, player_pos)
                
                pitch_ball = None
                if len(ball) > 0:
                    ball_pos = ball.get_anchors_coordinates(
                        sv.Position.BOTTOM_CENTER
                    )
                    pitch_ball = self.transform_points(self.current_H, ball_pos)
                
                birdseye = self.viz.create_birdseye(
                    pitch_pos, players.class_id, team_colors, pitch_ball
                )
            except Exception:
                birdseye = None
        
        return annotated, birdseye
    
    def process_video(self, video_path, progress_callback=None):
        """Process entire video file"""
        self.reset()
        self.train_teams(video_path, num_frames=30)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Get pitch dimensions once
        pitch_w, pitch_h = self.viz.get_pitch_size()
        
        out_a = tempfile.NamedTemporaryFile(suffix='_tracked.mp4', delete=False)
        out_b = tempfile.NamedTemporaryFile(suffix='_birdseye.mp4', delete=False)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        wa = cv2.VideoWriter(out_a.name, fourcc, fps, (width, height))
        wb = cv2.VideoWriter(out_b.name, fourcc, fps, (pitch_w, pitch_h))
        
        blank = self.viz.create_blank_pitch()
        fc = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated, birdseye = self.process_frame(frame, fc)
            
            # Write annotated frame
            if annotated.shape[0] != height or annotated.shape[1] != width:
                annotated = cv2.resize(annotated, (width, height))
            wa.write(annotated)
            
            # Write bird's eye view
            if birdseye is not None:
                if birdseye.shape[0] != pitch_h or birdseye.shape[1] != pitch_w:
                    birdseye = cv2.resize(birdseye, (pitch_w, pitch_h))
                wb.write(birdseye)
            else:
                wb.write(blank)
            
            fc += 1
            
            if progress_callback and fc % 5 == 0:
                progress_callback(fc / total_frames)
        
        cap.release()
        wa.release()
        wb.release()
        
        return out_a.name, out_b.name, fc
