from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        # convert the dataframe back to the list
        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]


        return ball_positions

    def get_ball_shot_frames(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        
        # midpoint of y
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2

        # rolling mean to smooth out small movements
        df_ball_positions['mid_y_smooth'] = df_ball_positions['mid_y'].rolling(window=3, min_periods=1).mean()

        # velocity (1st derivative) and acceleration (2nd derivative)
        df_ball_positions['velocity_y'] = df_ball_positions['mid_y_smooth'].diff()
        df_ball_positions['acceleration_y'] = df_ball_positions['velocity_y'].diff()

        # threshold based on acceleration magnitude (tweak this value!)
        acceleration_threshold = 10  # You might need to try 8, 12, 15, etc.

        # only count a hit if acceleration spike is large enough
        df_ball_positions['ball_hit'] = (df_ball_positions['acceleration_y'].abs() > acceleration_threshold).astype(int)
        
        # remove nearby duplicate frames
        raw_hits = df_ball_positions.index[df_ball_positions['ball_hit'] == 1].tolist()
        filtered_hits = []
        min_spacing = 20

        for hit in raw_hits:
            if not filtered_hits or hit - filtered_hits[-1] > min_spacing:
                filtered_hits.append(hit)

        return filtered_hits

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        # skip frame detection if we have a saved video of detection
        if read_from_stub is True and stub_path is not None:
            print("Detecting ball...")
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        # save output for reruns
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections

    def detect_frame(self,frame):
        results = self.model.predict(frame,conf=0.45)[0] 
        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result

        return ball_dict
    
    def draw_bboxes(self, video_frames, ball_detections):
        output_video_frames=[] 
        for frame, ball_dict in zip(video_frames, ball_detections):
            # draw bounding boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}", (int(bbox[0]), int(bbox[1] -10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2 )
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2) # 2 means only outside border of box
            output_video_frames.append(frame)


        return output_video_frames
