import cv2
import numpy as np
import supervision as sv
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.configs.soccer import SoccerPitchConfiguration


class PitchVisualizer:
    """Handles all drawing and visualization"""
    
    def __init__(self):
        self.config = SoccerPitchConfiguration()
        self.scale = 0.1
        self.padding = 50
        
    def annotate_frame(self, frame, detections, team_ids, tracker_ids, 
                       team_colors, ball_detections=None):
        """Draw jersey-colored ellipses on frame"""
        annotated = frame.copy()
        
        if len(detections) > 0:
            for bbox, team_id, tracker_id in zip(
                detections.xyxy, team_ids, tracker_ids
            ):
                x1, y1, x2, y2 = bbox.astype(int)
                
                # Force Python int from team_colors dict
                raw = team_colors.get(int(team_id), (128, 128, 128))
                color_b = int(raw[0])
                color_g = int(raw[1])
                color_r = int(raw[2])
                
                center_x = int((x1 + x2) / 2)
                center_y = int(y2)
                axis_x = int((x2 - x1) / 2)
                axis_y = int((y2 - y1) / 4)
                
                # Filled ellipse - MUST pass scalar values
                cv2.ellipse(
                    annotated,                          # image
                    (center_x, center_y),               # center
                    (axis_x, axis_y),                   # axes
                    0.0,                                # angle
                    0.0,                                # startAngle
                    360.0,                              # endAngle
                    (color_b, color_g, color_r),        # color
                    -1,                                 # thickness (filled)
                    cv2.LINE_AA                         # lineType
                )
                
                # Outline ellipse
                cv2.ellipse(
                    annotated,
                    (center_x, center_y),
                    (axis_x, axis_y),
                    0.0, 0.0, 360.0,
                    (0, 0, 0),                          # black outline
                    2,                                   # thickness
                    cv2.LINE_AA
                )
                
                # ID label
                text = f"#{tracker_id}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                tx = center_x - tw // 2
                ty = center_y - 15
                
                cv2.rectangle(
                    annotated,
                    (tx - 3, ty - th - 3),
                    (tx + tw + 3, ty + 3),
                    (0, 0, 0),
                    -1
                )
                cv2.putText(
                    annotated, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA
                )
        
        # Ball marker
        if ball_detections is not None and len(ball_detections) > 0:
            for bbox in ball_detections.xyxy:
                x1, y1, x2, y2 = bbox.astype(int)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cv2.drawMarker(
                    annotated, (cx, cy), (0, 255, 255),
                    cv2.MARKER_STAR, 15, 2, cv2.LINE_AA
                )
        
        return annotated
    
    def _make_pitch(self):
        """Create pitch image with consistent dimensions"""
        pitch = draw_pitch(
            config=self.config,
            padding=self.padding,
            scale=self.scale
        )
        return pitch
    
    def create_birdseye(self, player_positions, team_ids, team_colors, 
                        ball_position=None):
        """Create tactical bird's eye view"""
        pitch = self._make_pitch()
        
        if len(player_positions) > 0:
            for pos, team_id in zip(player_positions, team_ids):
                raw = team_colors.get(int(team_id), (128, 128, 128))
                c_r = int(raw[2])
                c_g = int(raw[1])
                c_b = int(raw[0])
                sv_color = sv.Color(r=c_r, g=c_g, b=c_b)
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
                face_color=sv.Color.from_hex('#FFFFFF'),
                edge_color=sv.Color.from_hex('#000000'),
                radius=8,
                pitch=pitch
            )
        
        return pitch
    
    def create_blank_pitch(self):
        """Create empty pitch"""
        return self._make_pitch()
    
    def get_pitch_size(self):
        """Get (width, height) of pitch image"""
        pitch = self._make_pitch()
        return pitch.shape[1], pitch.shape[0]
