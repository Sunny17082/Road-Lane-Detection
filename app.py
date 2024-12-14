import cv2 # type: ignore
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO

def dynamic_omega(image):
    """Calculate the dynamic omega value based on the image content."""
    # Calculate the average intensity of the image
    avg_intensity = np.mean(image)
    # Calculate the dynamic omega value
    omega = 0.95 - (avg_intensity / 255) * 0.5
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
    st.write(f"Dynamic Omega: {omega}")
    t = 1 - omega * (dark_channel_img / np.max(A))
    t = np.clip(t, 0.1, 1)
    t = cv2.bilateralFilter(t.astype(np.float32), 5, 0.1, 0.1)
    J = np.empty_like(I)
    for i in range(3):
        J[:, :, i] = (I[:, :, i] - A[i]) / t + A[i]
    J = np.clip(J, 0, 255).astype(np.uint8)
    return J, dark_channel_img, t

def average_lines(image, lines):
    """Average out multiple lines into a single line."""
    if len(lines) == 0:
        return None
    x_coords = []
    y_coords = []
    for line in lines:
        x_coords.extend([line[0], line[2]])
        y_coords.extend([line[1], line[3]])
    poly = np.polyfit(y_coords, x_coords, deg=1)
    slope = poly[0]
    intercept = poly[1]
    y1 = image.shape[0]
    y2 = int(y1 * 0.67)  # Approximate position for the lane
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
            # Calculate slope
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
            else:
                slope = np.inf  # Vertical line
            
            # Filter out vertical and near-horizontal lines
            if 0.43 < abs(slope) < 5:  # Adjust slope thresholds based on your use case
                if slope < 0:
                    left_lines.append((x1, y1, x2, y2))
                else:
                    right_lines.append((x1, y1, x2, y2))
    
    left_lane = average_lines(image, left_lines)
    right_lane = average_lines(image, right_lines)
    
    # Draw lanes on the image
    if left_lane is not None:
        cv2.line(line_image, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]), (0, 255, 0), 5)
    if right_lane is not None:
        cv2.line(line_image, (right_lane[0], right_lane[1]), (right_lane[2], right_lane[3]), (0, 255, 0), 5)
    
    return line_image


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
        (width, height // 1.7),
        (0, height // 1.7)
    ], np.int32)
    cv2.fillPoly(mask, [polygon], 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    line_image = detect_lines(cropped_edges, image)
    lane_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    return lane_image, gray, blur, edges, cropped_edges



st.title("Foggy Lane Detection")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "webp"])
if uploaded_file is not None:
    # Load the image
    input_image = np.array(Image.open(uploaded_file))
    st.image(input_image, caption="Uploaded Image", use_container_width=True)

    # Step 1: Defog the image
    defogged_image, dark_channel_img, transmission_map = defog_image(input_image)
    st.image(defogged_image, caption="Defogged Image", use_container_width=True)
    st.image(dark_channel_img, caption="Dark Channel", use_container_width=True, clamp=True)
    st.image(transmission_map, caption="Transmission Map", use_container_width=True, clamp=True)

    # Step 2: Detect lanes
    lane_image, gray, blur, edges, cropped_edges = detect_lanes(defogged_image)
    st.image(gray, caption="Grayscale Image", use_container_width=True, clamp=True)
    st.image(blur, caption="Blurred Image", use_container_width=True, clamp=True)
    st.image(edges, caption="Edges", use_container_width=True, clamp=True)
    st.image(cropped_edges, caption="Region of Interest", use_container_width=True, clamp=True)
    st.image(lane_image, caption="Final Lane Detection", use_container_width=True)

    # Convert the final image to a downloadable format
    lane_image_pil = Image.fromarray(lane_image.astype('uint8'))  # Convert to PIL Image
    buffer = BytesIO()
    lane_image_pil.save(buffer, format="PNG")  # Save as PNG
    buffer.seek(0)

    # Add download button
    st.download_button(
        label="Download Final Lane Detection Image",
        data=buffer,
        file_name="lane_detection_result.png",
        mime="image/png"
    )