import cv2
import numpy as np
import math
from typing import Optional, Tuple, List


def canny_edge(image: np.ndarray) -> np.ndarray:
    """Convert to gray, blur, and apply Canny edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges


def region_of_interest(edges: np.ndarray) -> np.ndarray:
    """
    Keep only the lower road area using a trapezoidal ROI mask.
    Input should be a single-channel edge image.
    """
    height, width = edges.shape[:2]

    # Trapezoid roughly covering the lane area ahead
    vertices = np.array([[
        (int(width * 0.10), height),          # bottom-left
        (int(width * 0.90), height),          # bottom-right
        (int(width * 0.60), int(height * 0.60)),  # top-right
        (int(width * 0.40), int(height * 0.60)),  # top-left
    ]], dtype=np.int32)

    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(edges, mask)


def average_slope_intercept(lines: Optional[np.ndarray]) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Average Hough line segments into left and right lane (slope, intercept)."""
    if lines is None:
        return None, None

    left_fit: List[Tuple[float, float]] = []
    right_fit: List[Tuple[float, float]] = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        # Skip degenerate / near-vertical segments that make polyfit unstable
        if x2 == x1:
            continue

        # Fit y = m x + b
        slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)

        # Filter out near-horizontal noise
        if abs(slope) < 0.3:
            continue

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_avg = tuple(np.mean(left_fit, axis=0)) if left_fit else None
    right_avg = tuple(np.mean(right_fit, axis=0)) if right_fit else None
    return left_avg, right_avg


def make_line_points(y_bottom: int, y_top: int, line_params: Optional[Tuple[float, float]]) -> Optional[List[Tuple[int, int]]]:
    """Convert (slope, intercept) to two points [(x_bottom, y_bottom), (x_top, y_top)]."""
    if line_params is None:
        return None

    slope, intercept = line_params

    # Avoid division blow-ups
    if abs(slope) < 1e-6:
        return None

    x_bottom = int((y_bottom - intercept) / slope)
    x_top = int((y_top - intercept) / slope)
    return [(x_bottom, y_bottom), (x_top, y_top)]


def draw_lane(image: np.ndarray, lines: List[Optional[List[Tuple[int, int]]]],
              color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 5) -> np.ndarray:
    """Draw lane lines (if present) onto a blank image."""
    line_image = np.zeros_like(image)

    for pts in lines:
        if pts is None:
            continue
        (x1, y1), (x2, y2) = pts
        cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)

    return line_image


def compute_steering(left_line: Optional[List[Tuple[int, int]]],
                     right_line: Optional[List[Tuple[int, int]]],
                     width: int) -> float:
    """
    Estimate steering angle (degrees) by comparing lane center at the bottom of the frame
    to the image center. Positive means steer right, negative steer left.
    """
    if left_line is None or right_line is None:
        return 0.0

    # Use bottom x-coordinates for a consistent "near car" lane center estimate
    left_bottom_x = left_line[0][0]
    right_bottom_x = right_line[0][0]

    lane_center_x = 0.5 * (left_bottom_x + right_bottom_x)
    car_center_x = 0.5 * width
    offset = lane_center_x - car_center_x

    # Scale by half the image width to get a normalized-ish angle
    steering_angle = math.degrees(math.atan(offset / (width * 0.5)))
    return steering_angle


def main(video_path="road_video.mp4"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {video_path}")

    steering_smoothed = 0.0
    alpha = 0.2  # smoothing factor (0=no update, 1=no smoothing)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]

        edges = canny_edge(frame)
        roi = region_of_interest(edges)

        lines = cv2.HoughLinesP(
            roi,
            rho=2,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=40,
            maxLineGap=100,
        )

        left_avg, right_avg = average_slope_intercept(lines)

        y_bottom, y_top = height, int(height * 0.6)
        left_pts = make_line_points(y_bottom, y_top, left_avg)
        right_pts = make_line_points(y_bottom, y_top, right_avg)

        lane_img = draw_lane(frame, [left_pts, right_pts])

        steering = compute_steering(left_pts, right_pts, width)
        steering_smoothed = (1 - alpha) * steering_smoothed + alpha * steering

        overlay = cv2.addWeighted(frame, 0.8, lane_img, 1.0, 0)
        cv2.putText(
            overlay,
            f"Steer: {steering_smoothed:.1f}°",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        cv2.imshow("Lane Follower", overlay)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Replace the string below with your video file or 0 for webcam
    main("/Users/tosinkuye/Downloads/11933881_2160_3840_30fps.mp4")
