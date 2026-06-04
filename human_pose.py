import mediapipe as mp
import cv2
import numpy as np
import os

try:
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
except AttributeError:
    from mediapipe.python import solutions as mp_solutions

    mp_drawing = mp_solutions.drawing_utils
    mp_pose = mp_solutions.pose

pose = mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        enable_segmentation=True)

def get_human_pose(frame):
    input_image = cv2.resize(frame, (60, 60))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    return pose.process(input_image)

def display_human_pose(pose_results, width, height):
    dots = np.zeros((width, height), dtype=np.uint8)
    if pose_results.segmentation_mask is not None:
        dots = cv2.resize(pose_results.segmentation_mask, (width, height), cv2.INTER_AREA)
        dots = (dots > 0.5).astype(np.uint8)
        dots = np.fliplr(dots)
    
    return dots

def eyes_visible_and_facing_camera(pose_results):
    # Check if all relevant landmarks are detected with reasonable confidence
    # - Left eye: 2 (inner), 3 (outer)
    # - Right eye: 5 (inner), 4 (outer)
    if pose_results.pose_landmarks is None or pose_results.pose_world_landmarks is None:
        return False, "Pose landmarks not detected", None

    landmarks = pose_results.pose_landmarks.landmark
    world_landmarks = pose_results.pose_world_landmarks.landmark
    eye_landmarks = [landmarks[0], landmarks[2], landmarks[3], landmarks[4], landmarks[5]]
    
    # Visibility threshold
    confidence_threshold = 0.7
    
    # Check if all eye landmarks are visible with high confidence
    all_eye_landmarks_visible = all(landmark.visibility > confidence_threshold for landmark in eye_landmarks)
    if not all_eye_landmarks_visible:
        return False, "Eye landmarks not clearly visible", None
    
    # Calculate the midpoint between the eyes in 3D space
    left_eye_inner_3d = np.array([
        world_landmarks[2].x,
        world_landmarks[2].y,
        world_landmarks[2].z
    ])
    
    right_eye_inner_3d = np.array([
        world_landmarks[5].x,
        world_landmarks[5].y,
        world_landmarks[5].z
    ])
    
    face_direction = right_eye_inner_3d - left_eye_inner_3d
    # Discard the y-component to focus on the horizontal planea and rotate the vector 90 degrees
    face_direction = np.array([face_direction[2], 0, -face_direction[0]])  # Rotate 90 degrees around y-axis
    
    # Calculate the camera direction vector (pointing straight into the camera)
    # In MediaPipe's coordinate system, the z-axis points toward the camera
    camera_direction = np.array([0, 0, 1])
    
    # Calculate the angle between face direction and camera direction
    # First normalize the vectors
    face_direction_normalized = face_direction / np.linalg.norm(face_direction)
    
    # Calculate the dot product and then the angle
    dot_product = np.dot(face_direction_normalized, camera_direction)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    # If the person is facing the camera, the the z-component of the face direction should be positive
    facing_forward = face_direction[2] > 0
    
    # Define threshold for the angle
    max_angle_threshold = 30  # degrees
    
    # Check if the face is looking at the camera within the threshold
    facing_camera = facing_forward and angle_deg < max_angle_threshold
    
    if facing_camera:
        return True, f"Facing camera: {angle_deg:.1f}°", angle_deg
    else:
        if not facing_forward:
            return False, f"Face turned away from camera: {angle_deg:.1f}°", angle_deg
        else:
            return False, f"Not directly facing camera: {angle_deg:.1f}°", angle_deg

def estimate_distance(pose_results):
    FOCAL_SCALE = float(os.getenv('FOCAL_SCALE', '1.0'))

    # Estimate distance based on the size of the person in the frame
    if pose_results.pose_landmarks is None:
        return None, []

    KNOWN_DISTANCES = [
        {"landmark0": 11, "landmark1": 12, "value": 0.45, "name": "shoulder width"},
        {"landmark0": 23, "landmark1": 24, "value": 0.37, "name": "hip width"},
        {"landmark0": 11, "landmark1": 23, "value": 0.4, "name": "left shoulder to hip"},
        {"landmark0": 12, "landmark1": 24, "value": 0.4, "name": "right shoulder to hip"},
        {"landmark0": 2, "landmark1": 5, "value": 0.06, "name": "eye distance"},
        {"landmark0": 7, "landmark1": 8, "value": 0.14, "name": "ear distance"},
    ]   
    landmarks = pose_results.pose_landmarks.landmark

    distance_estimates = []
    for known_distance in KNOWN_DISTANCES:
        landmark0 = landmarks[known_distance["landmark0"]]
        landmark1 = landmarks[known_distance["landmark1"]]

        # Check if both landmarks are visible
        if landmark0.visibility < 0.5 or landmark1.visibility < 0.5:
            continue

        # Calculate the distance between the two landmarks
        point0 = np.array([landmark0.x, landmark0.y, landmark0.z])
        point1 = np.array([landmark1.x, landmark1.y, landmark1.z])
        distance = np.linalg.norm(point0 - point1)

        # Discard distances with large z component
        # because they are likely not very accurate
        #z_distance = abs(point0[2] - point1[2])
        #if z_distance > 0.5 * distance:
        #    continue

        distance_estimates.append(((known_distance["value"] * FOCAL_SCALE) / distance, distance, known_distance["name"]))


    if len(distance_estimates) == 0:
        return None, []

    return sum(value for value, _, _ in distance_estimates) / len(distance_estimates), distance_estimates

def get_right_index_finger_position(pose_results):
    if pose_results.pose_landmarks is None:
        return None, None

    landmarks = pose_results.pose_landmarks.landmark
    # Get right index finger if it is clearly visible otherwise use right thumb if it is more visible otherwise estimate from both
    right_index = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX]
    right_thumb = landmarks[mp_pose.PoseLandmark.RIGHT_THUMB]
    if right_index.visibility > 0.7:
        return right_index.x, right_index.y
    elif right_thumb.visibility > 0.7:
        return right_thumb.x, right_thumb.y
    elif right_index.visibility > 0.3 and right_thumb.visibility > 0.3:
        # Estimate position as average of both
        x = (right_index.x + right_thumb.x) / 2
        y = (right_index.y + right_thumb.y) / 2
        return x, y
    else:
        return None, None


def is_right_index_in_top_right_corner(pose_results):
    x, y = get_right_index_finger_position(pose_results)

    # Check if the index finger is in the top right corner (e.g., top 20% and right 20%)
    if x is not None and y is not None and x < 0.2 and y < 0.2:
        return True, x, y

    return False, x, y

def draw_right_index_pointer(frame, pose_results, size=1):
    finger_x, finger_y = get_right_index_finger_position(pose_results)
    height, width = frame.shape

    if finger_x is not None and finger_y is not None:
        x = int(width - (finger_x * width))
        y = int(finger_y * height)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), size//2, 0, -1)
            cv2.circle(frame, (x, y), size//2, 1, 1)

    return frame
