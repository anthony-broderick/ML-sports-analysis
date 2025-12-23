from utils import (read_video,
                   save_video)
from trackers import PlayerTracker,BallTracker
from pathlib import Path
from court_keypoint_placement import CourtKeypointsManager
from mini_court import MiniCourt
from datetime import datetime
import cv2

# set to true if you want to replace the stubs
read_new_player_stub = False 
read_new_ball_stub = False

def main():
    # read video
    input_video_path = "input_videos/input_video.mp4"
    video_frames, fps = read_video(input_video_path)

    # detect players and ball
    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path='models/ball_tracker_2.pt')

    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=read_new_player_stub,
                                                     stub_path="tracker_stubs/player_detections.pkl"
                                                     )
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=read_new_ball_stub,
                                                     stub_path="tracker_stubs/ball_detections.pkl"
                                                     )
    print("Interpolating ball positions...")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # output txt file for court keypoints
    keypoints_output = "court_keypoint_placement/court_keypoints.txt"
    keypoint_stub = True # set to false if you want to replace keypoints

    keypoint_manager = CourtKeypointsManager()
    if keypoint_stub == False:
        print("Choosing court keypoints...")
        keypoint_manager.click_and_save_keypoints_from_video(input_video_path, keypoints_output)
    court_keypoints = keypoint_manager.load_keypoints(keypoints_output)

    # choose players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # MiniCourt
    mini_court = MiniCourt(video_frames[0])

    # detect ball shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    # print("Ball shot frames:", ball_shot_frames)
    
    # convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_box_to_mini_court_coordinates(player_detections, 
                                                                                                                       ball_detections, 
                                                                                                                       court_keypoints
                                                                                                                       )
    

    # draw output
    # draw player bounding boxes
    print("Drawing bounding boxes...")
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    print("Drawing ball bounding boxes...")
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    print("Drawing court keypoints...")
    for i in range(len(output_video_frames)):
        keypoint_manager.draw_keypoints(output_video_frames[i], court_keypoints)
        

    timestamp = datetime.now().strftime("%m_%d_%H-%M")  # e.g., "05_17_14-22"
    filename = f"output_videos/{timestamp}.avi"

    # draw Mini Court
    print("Drawing mini court...")
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    print("Drawing player and ball points on mini court...")
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0,255,255))

    print("Adding frame numbers to video...")
    # draw frame number on top left corner of video
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    print("Saving video...")
    save_video(output_video_frames, filename, fps)

if __name__ == "__main__":
    main()