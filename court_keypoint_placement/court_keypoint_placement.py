import cv2
import os

class CourtKeypointsManager:
    def __init__(self):
        self.points = []

    def _click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked: ({x}, {y})")
            self.points.append((x, y))
            cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Click Court Points", param)

    def extract_first_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Failed to read first frame from video.")
        return frame

    def click_and_save_keypoints_from_frame(self, frame, output_file):
        self.points = []
        clone = frame.copy()

        cv2.imshow("Click Court Points", clone)
        cv2.setMouseCallback("Click Court Points", self._click_event, clone)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        with open(output_file, "w") as f:
            for p in self.points:
                f.write(f"{p[0]},{p[1]}\n")
        print(f"Saved {len(self.points)} keypoints to {output_file}")

    def click_and_save_keypoints_from_video(self, video_path, output_file):
        frame = self.extract_first_frame(video_path)
        self.click_and_save_keypoints_from_frame(frame, output_file)

    def load_keypoints(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Keypoint file not found at: {path}")
        with open(path, "r") as f:
            return [tuple(map(int, line.strip().split(","))) for line in f.readlines()]

    def draw_keypoints(self, image, keypoints):
        for (x, y) in keypoints:
            cv2.circle(image, (x, y), 8, (0, 255, 0), -1)