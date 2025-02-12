import cv2 as cv
import numpy as np
import os
from PIL import Image
import glob
import matplotlib.pyplot as plt

def frames_from_video(video_path, save_path, extraction_rate=1):
    """
    Extract frames from a video file and save them.

    Args:
        video_path (str): Path to the video file.
        save_path (str): Directory to save extracted frames.
        extraction_rate (int): Extract one frame every `extraction_rate` seconds.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cap = cv.VideoCapture(video_path)
    fps = int(cap.get(cv.CAP_PROP_FPS))
    frame_interval = fps * extraction_rate
    
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_name = os.path.join(save_path, f"frame_{saved_count}.png")
            cv.imwrite(frame_name, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Frames saved in {save_path}")


def optical_flow(frames_saved_path):
    """
    Compute optical flow using Lucas-Kanade method.

    Args:
        frames_saved_path (str): Path to saved frames.
    """
    frames_path = sorted(glob.glob(os.path.join(frames_saved_path, '*.png')))
    frames_images = [cv.imread(frame) for frame in frames_path]

    # Shi-Tomasi corner detection parameters
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Lucas-Kanade optical flow parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # Random colors for visualization
    color = np.random.randint(0, 255, (100, 3))

    # First frame processing
    first_frame = frames_images[0]
    first_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(first_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(first_frame)

    # Ensure tracking starts immediately by initializing points with the first frame
    if p0 is None:
        print("No features detected in the first frame. Initializing random points.")
        h, w = first_gray.shape
        p0 = np.array([[[x, y]] for y in range(0, h, h // 10) for x in range(0, w, w // 10)], dtype=np.float32)

    for i in range(1, len(frames_images)):
        frame = frames_images[i]
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(first_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Draw the tracks
            for j, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[j].tolist(), 2)
                frame = cv.circle(frame, (int(a), int(b)), 5, color[j].tolist(), -1)

            output = cv.add(frame, mask)
            cv.imshow('Optical Flow', output)

            # Save the resulting frame with tracks
            cv.imwrite(os.path.join(frames_saved_path, f"optical_flow_{i}.png"), output)

            if cv.waitKey(30) & 0xFF == 27:
                break

            first_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

    cv.destroyAllWindows()


def create_video_from_frames(frames_path, output_video_path):
    """
    Create a video from saved frames.

    Args:
        frames_path (str): Path to saved frames.
        output_video_path (str): Path to save the output video.
    """
    frames = sorted(glob.glob(os.path.join(frames_path, '*.png')))
    
    if len(frames) == 0:
        print("No frames to create a video.")
        return

    frame = cv.imread(frames[0])
    height, width, layers = frame.shape
    
    video = cv.VideoWriter(output_video_path, cv.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

    for frame_file in frames:
        frame = cv.imread(frame_file)
        video.write(frame)

    video.release()
    print(f"Video saved at {output_video_path}")

# Example usage
video_path = "path_to_your_video.mp4"
save_frames_path = "frames"
output_video_path = "output_video.mp4"

frames_from_video(video_path, save_frames_path)
optical_flow(save_frames_path)
create_video_from_frames(save_frames_path, output_video_path)
