import cv2
import numpy as np
import supervision as sv
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.configs.soccer import SoccerPitchConfiguration

class PitchVisualizer:
    """Handles all drawing and visualization"""
    
    def __init__(self):
        self.config = SoccerPitchConfiguration()
        
    def annotate_frame(self, frame, detections, team_ids, tracker_ids, 
                       team_colors, ball_detections=None):
        """Draw jersey-colored ellipses on frame"""
        annotated = frame.copy()
        
        if len(detections) > 0:
            for bbox, team_id, tracker_id in zip(
                detections.xyxy, team_ids, tracker_ids
            ):
                x1, y1, x2, y2 = bbox.astype(int)
                
                # Get color and ENSURE it's plain Python ints
                raw_color = team_colors.get(int(team_id), (128, 128, 128))
                color = (int(raw_color[0]), int(raw_color[1]), int(raw_color[2]))
                
                # Ellipse under player
                center = (int((x1 + x2) / 2), int(y2))
                axes = (int((x2 - x1) / 2), int((y2 - y1) / 4))
                
                # FIXED: Use separate calls instead of overloaded syntax
                cv2.ellipse(annotated, center, axes, 0, 0, 360, color, -1, cv2.LINE_AA)
                cv2.ellipse(annotated, center, axes, 0, 0, 360, (0, 0, 0), 2, cv2.LINE_AA)
                
                # ID label
                text = f"#{tracker_id}"
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                tx, ty = center[0] - tw//2, center[1] - 15
                
                cv2.rectangle(annotated, 
                            (tx-3, ty-th-3), (tx+tw+3, ty+3),
                            (0, 0, 0), -1)
                cv2.putText(annotated, text, (tx, ty),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                          (255, 255, 255), 2, cv2.LINE_AA)
        
        # Ball marker
        if ball_detections is not None and len(ball_detections) > 0:
            for bbox in ball_detections.xyxy:
                x1, y1, x2, y2 = bbox.astype(int)
                center = (int((x1+x2)/2), int((y1+y2)/2))
                cv2.drawMarker(annotated, center, (0, 255, 255),
                             cv2.MARKER_STAR, 15, 2, cv2.LINE_AA)
        
        return annotated
    
    def create_birdseye(self, player_positions, team_ids, team_colors, 
                        ball_position=None):
        """Create tactical bird's eye view"""
        pitch = draw_pitch(config=self.config, padding=50, scale=0.1)
        
        if len(player_positions) > 0:
            for pos, team_id in zip(player_positions, team_ids):
                raw_color = team_colors.get(int(team_id), (128, 128, 128))
                color_bgr = (int(raw_color[0]), int(raw_color[1]), int(raw_color[2]))
                sv_color = sv.Color(
                    r=color_bgr[2], g=color_bgr[1], b=color_bgr[0]
                )
                pitch = draw_points_on_pitch(
                    config=self.config,
                    xy=np.array([pos]),
                    face_color=sv_color,
                    edge_color=sv.Color.BLACK,
                    radius=12,
                    pitch=pitch
                )
        
        if ball_position is not None and len(ball_position) > 0:
            pitch = draw_points_on_pitch(
                config=self.config,
                xy=ball_position,
                face_color=sv.Color.WHITE,
                edge_color=sv.Color.BLACK,
                radius=8,
                pitch=pitch
            )
        
        return pitch
    
    def create_blank_pitch(self):
        """Create empty pitch for when homography fails"""
        return draw_pitch(config=self.config, padding=50, scale=0.1)
