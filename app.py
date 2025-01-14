import json
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO

def dynamic_omega(image):
    """Calculate the dynamic omega value based on the image content."""
    avg_intensity = np.mean(image)
    if avg_intensity < 90:
        omega = 0.75
    else:
        omega = 0.95
    return omega

def dark_channel(image, size=5):
    """Calculate the dark channel prior of an image."""
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

def estimate_atmospheric_light(image, dark_channel):
    """Estimate atmospheric light based on the dark channel."""
    h, w = dark_channel.shape
    num_pixels = h * w
    num_brightest_pixels = int(max(num_pixels // 1000, 1))
    indices = np.unravel_index(np.argsort(dark_channel.ravel())[-num_brightest_pixels:], dark_channel.shape)
    A = np.mean(image[indices], axis=0)
    return A

def defog_image(image):
    """Defog the input image using improved dark channel prior."""
    I = (image).astype(np.uint8)
    dark_channel_img = dark_channel(I)
    A = estimate_atmospheric_light(I, dark_channel_img)
    omega = dynamic_omega(I)
    t = 1 - omega * (dark_channel_img / np.max(A))
    t = np.clip(t, 0.1, 1)
    t = cv2.bilateralFilter(t.astype(np.float32), 5, 0.1, 0.1)
    J = np.empty_like(I)
    for i in range(3):
        J[:, :, i] = (I[:, :, i] - A[i]) / t + A[i]
    J = np.clip(J, 0, 255).astype(np.uint8)
    return J, dark_channel_img, t

def average_lines(image, lines, slope_threshold=0.1):
    """Average out multiple lines into a single line with outlier filtering."""
    if len(lines) == 0:
        return None

    # Separate x and y coordinates for all line segments
    x_coords, y_coords, weights = [], [], []

    for x1, y1, x2, y2 in lines:
        # Calculate slope and line length
        dx, dy = x2 - x1, y2 - y1
        slope = dy / dx if dx != 0 else np.inf
        length = np.sqrt(dx**2 + dy**2)

        # Filter lines based on slope threshold
        if -slope_threshold < slope < slope_threshold:
            continue

        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])
        weights.extend([length, length])  # Weight by line length

    if len(x_coords) == 0:
        return None

    # Fit a weighted least-squares line to the points
    poly = np.polyfit(y_coords, x_coords, deg=1, w=weights)
    slope, intercept = poly[0], poly[1]

    # Define points for the averaged line
    y1 = image.shape[0]  # Bottom of the image
    y2 = int(y1 * 0.67)  # 2/3 height of the image
    x1 = int(slope * y1 + intercept)
    x2 = int(slope * y2 + intercept)

    return (x1, y1, x2, y2)

def detect_lines(cropped_edges, image):
    """Detect and filter lanes from the cropped edges image."""
    lines = cv2.HoughLinesP(
        cropped_edges, rho=1, theta=np.pi / 180, threshold=20, minLineLength=50, maxLineGap=150
    )
    line_image = np.zeros_like(image)

    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
            else:
                slope = np.inf

            if 0.43 < abs(slope) < 5:
                if slope < 0:
                    left_lines.append((x1, y1, x2, y2))
                else:
                    right_lines.append((x1, y1, x2, y2))

    left_lane = average_lines(image, left_lines)
    right_lane = average_lines(image, right_lines)

    if left_lane is not None:
        cv2.line(line_image, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]), (0, 255, 0), 5)
    if right_lane is not None:
        cv2.line(line_image, (right_lane[0], right_lane[1]), (right_lane[2], right_lane[3]), (0, 255, 0), 5)

    return line_image, left_lane, right_lane

def detect_lanes(image):
    """Detect lanes in a defogged image while filtering out vertical lines."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([
        (0, height),
        (width, height),
        (width, height // 1.9),
        (0, height // 1.9)
    ], np.int32)
    cv2.fillPoly(mask, [polygon], 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    line_image, left_lane, right_lane = detect_lines(cropped_edges, image)
    lane_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    return lane_image, gray, blur, edges, cropped_edges, left_lane, right_lane

def calculate_iou(detected_lane, ground_truth_polygon, image_shape):
    """Calculate Intersection over Union (IoU) between a detected lane and the ground truth polygon."""
    # Create the lane mask based on the input image's shape
    lane_mask = np.zeros(image_shape[:2], dtype=np.uint8)  # Use the input image's height and width
    cv2.line(lane_mask, (detected_lane[0], detected_lane[1]), (detected_lane[2], detected_lane[3]), 255, 5)

    # Convert ground truth polygon to a mask
    gt_polygon = np.array([[(int(point['x']), int(point['y'])) for point in ground_truth_polygon]])
    gt_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(gt_mask, gt_polygon, 255)

    # Ensure both masks are the same size before performing logical operations
    if lane_mask.shape != gt_mask.shape:
        gt_mask = cv2.resize(gt_mask, (lane_mask.shape[1], lane_mask.shape[0]))

    intersection = np.logical_and(lane_mask, gt_mask).sum()
    union = np.logical_or(lane_mask, gt_mask).sum()

    return intersection / union if union > 0 else 0

def downgrade_image(image, uploaded_file, max_file_size_kb=500, resolution=(640, 360), quality=50):
    """
    Resize and compress the image only if its size exceeds the specified limit (in KB).
    :param image: The input image as a NumPy array.
    :param uploaded_file: The uploaded file to check size.
    :param max_file_size_kb: Maximum file size allowed in KB.
    :param resolution: The target resolution for resizing.
    :param quality: The target quality for compression (1-100).
    :return: The processed image (if needed) or the original image.
    """
    # Check the file size
    file_size_kb = len(uploaded_file.getvalue()) / 1024  # Convert bytes to KB

    # Process only if the file size is greater than the specified limit
    if file_size_kb > max_file_size_kb:
        # Check if the image has an alpha channel and convert it to RGB if necessary
        if image.shape[-1] == 4:  # Check if the image is RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize the image
        resized_image = cv2.resize(image, resolution, interpolation=cv2.INTER_AREA)
        
        # Convert to PIL for quality adjustment
        pil_image = Image.fromarray(resized_image)
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=quality)
        downgraded_image = np.array(Image.open(buffer))
        
        return downgraded_image
    else:
        # Return the original image if the size is below the limit
        return image



def compute_lane_accuracy(left_lane, right_lane, ground_truth_json, image_shape):
    """Compute the accuracy of lane detection using IoU."""
    gt_data = json.load(ground_truth_json)

    left_gt_polygon = gt_data[0]["content"]
    right_gt_polygon = gt_data[1]["content"]

    left_iou = calculate_iou(left_lane, left_gt_polygon, image_shape) if left_lane is not None else 0
    right_iou = calculate_iou(right_lane, right_gt_polygon, image_shape) if right_lane is not None else 0
    st.write(f"Left Lane IoU: {left_iou:.2f}, Right Lane IoU: {right_iou:.2f}")
    return (left_iou + right_iou) / 2

st.title("Foggy Lane Detection with Accuracy")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "webp", "avif"])
ground_truth_file = st.file_uploader("Upload Ground Truth", type=["json"])

if uploaded_file is not None:
    input_image = np.array(Image.open(uploaded_file))
    st.image(input_image, caption="Uploaded Image", use_container_width=True)
    
    # Downgrade the image
    downgraded_image = downgrade_image(input_image, uploaded_file)
    st.image(downgraded_image, caption="Downgraded Image", use_container_width=True)

    # Process the downgraded image
    defogged_image, dark_channel_img, transmission_map = defog_image(downgraded_image)
    st.image(dark_channel_img, caption="Dark Channel Image", use_container_width=True)
    st.image(transmission_map, caption="Transmission Map", use_container_width=True)
    st.image(defogged_image, caption="Defogged Image", use_container_width=True)
    
    lane_image, gray, blur, edges, cropped_edges, left_lane, right_lane = detect_lanes(defogged_image)
    st.image(gray, caption="Gray Image", use_container_width=True)
    st.image(blur, caption="Blurred Image", use_container_width=True)
    st.image(edges, caption="Edges Image", use_container_width=True)
    st.image(cropped_edges, caption="Cropped Edges Image", use_container_width=True)
    st.image(lane_image, caption="Final Lane Detection", use_container_width=True)

if ground_truth_file is not None:
    accuracy = compute_lane_accuracy(left_lane, right_lane, ground_truth_file, input_image.shape)
    st.write(f"Lane Detection Accuracy (IoU): {accuracy * 100:.2f}%")
