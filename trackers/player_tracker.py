from ultralytics import YOLO
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_players = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_players}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    # choose_players function made with help of ChatGPT
    def choose_players(self, court_keypoints, player_dict):
        # Step 1: Compute the centroid of the court
        x_coords = [p[0] for p in court_keypoints]
        y_coords = [p[1] for p in court_keypoints]
        court_center = (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

        
        # Step 2: Measure each player's distance to the court center
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)
            distance = measure_distance(player_center, court_center)
            distances.append((track_id, distance))
        
        # Step 3: Sort by distance and choose the closest 4
        distances.sort(key=lambda x: x[1])
        chosen_players = [distances[i][0] for i in range(min(4, len(distances)))]
        return chosen_players
            

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        # skip frame detection if we have a saved video of detection
        if read_from_stub is True and stub_path is not None:
            print("Detecting players...")
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        # save output for reruns
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections

    def detect_frame(self,frame):
        results = self.model.track(frame, persist=True)[0] # persist tells it that there are multiple frames, remembering tracks before
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result

        return player_dict
    
    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames=[]
        for frame, player_dict in zip(video_frames, player_detections):
            # draw bounding boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1] -10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2 )
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 225), 2) # 2 means only outside border of box
            output_video_frames.append(frame)

        return output_video_frames
