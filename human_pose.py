import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        enable_segmentation=True)

def get_human_pose(frame):
    input_image = cv2.resize(frame, (80, 60))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    return pose.process(input_image)

def display_human_pose(pose_results, width, height):
    dots = np.zeros((width, height), dtype=np.uint8)
    if pose_results.segmentation_mask is not None:
        dots = cv2.resize(pose_results.segmentation_mask, (width, height), cv2.INTER_AREA)
        dots = (dots > 0.5).astype(np.uint8)
        dots = np.fliplr(dots)
    
    return dots

def eyes_visible(pose_results):
    if pose_results.pose_landmarks:
        left_eye = pose_results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
        right_eye = pose_results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]
        print(f"Left eye: {left_eye}")
        print(f"Right eye: {right_eye}")
        return left_eye.visibility > 0.5 and right_eye.visibility > 0.5
    return False

def check_eyes_visible_and_facing_camera(pose_results):
    # Check if all relevant landmarks are detected with reasonable confidence
    # - Nose: 0
    # - Left eye: 2 (inner), 3 (outer)
    # - Right eye: 5 (inner), 4 (outer)

    landmarks = pose_results.pose_landmarks.landmark
    world_landmarks = pose_results.pose_world_landmarks.landmark
    eye_landmarks = [landmarks[0], landmarks[2], landmarks[3], landmarks[4], landmarks[5]]
    
    # Visibility threshold
    confidence_threshold = 0.7
    
    # Check if all eye landmarks are visible with high confidence
    all_eye_landmarks_visible = all(landmark.visibility > confidence_threshold for landmark in eye_landmarks)
    if not all_eye_landmarks_visible:
        return False, "Eye landmarks not clearly visible"
    
    # Use world landmarks for direction calculation
    # Get 3D coordinates (x, y, z) for nose and eyes
    nose_3d = np.array([
        world_landmarks[0].x,
        world_landmarks[0].y,
        world_landmarks[0].z
    ])
    
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
    
    eyes_midpoint_3d = (left_eye_inner_3d + right_eye_inner_3d) / 2
    
    # Calculate the face direction vector from eyes midpoint to nose
    face_direction = nose_3d - eyes_midpoint_3d
    
    # Calculate the camera direction vector (pointing straight into the camera)
    # In MediaPipe's coordinate system, the z-axis points toward the camera
    camera_direction = np.array([0, 0, 1])
    
    # Calculate the angle between face direction and camera direction
    # First normalize the vectors
    face_direction_normalized = face_direction / np.linalg.norm(face_direction)
    print(face_direction_normalized)
    
    # Calculate the dot product and then the angle
    dot_product = np.dot(face_direction_normalized, camera_direction)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    # If the person is facing the camera, the nose should be in front of the eyes
    # which means the z-component of the face direction should be positive
    facing_forward = face_direction[2] > 0
    
    # Define threshold for the angle
    max_angle_threshold = 30  # degrees
    
    # Check if the face is looking at the camera within the threshold
    facing_camera = facing_forward and angle_deg < max_angle_threshold
    
    # Optional: Calculate left-right head rotation
    # Project the face direction onto the x-z plane
    horizontal_direction = np.array([face_direction[0], 0, face_direction[2]])
    horizontal_direction_normalized = horizontal_direction / np.linalg.norm(horizontal_direction)
    
    # Calculate horizontal angle
    horizontal_angle_rad = np.arccos(np.clip(horizontal_direction_normalized[2], -1.0, 1.0))
    horizontal_angle_deg = np.degrees(horizontal_angle_rad)
    if horizontal_direction_normalized[0] < 0:
        horizontal_angle_deg = -horizontal_angle_deg  # Negative for looking left
    
    # Calculate up-down head tilt
    vertical_direction = np.array([0, face_direction[1], face_direction[2]])
    vertical_direction_normalized = vertical_direction / np.linalg.norm(vertical_direction)
    
    # Calculate vertical angle
    vertical_angle_rad = np.arccos(np.clip(vertical_direction_normalized[2], -1.0, 1.0))
    vertical_angle_deg = np.degrees(vertical_angle_rad)
    if vertical_direction_normalized[1] < 0:
        vertical_angle_deg = -vertical_angle_deg  # Negative for looking up
    
    if facing_camera:
        return True, f"Facing camera: {angle_deg:.1f}°, H: {horizontal_angle_deg:.1f}°, V: {vertical_angle_deg:.1f}°"
    else:
        if not facing_forward:
            return False, f"Face turned away from camera: {angle_deg:.1f}°"
        else:
            return False, f"Not directly facing camera: {angle_deg:.1f}°, H: {horizontal_angle_deg:.1f}°, V: {vertical_angle_deg:.1f}°"