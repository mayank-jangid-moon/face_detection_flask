import cv2
import numpy as np
from mtcnn import MTCNN
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Face detection class using MTCNN (Multi-task Cascaded Convolutional Networks).
    
    MTCNN is a deep learning framework for face detection and alignment,
    which performs face detection and facial landmark detection jointly.
    """
    
    def __init__(self, min_face_size=20, scale_factor=0.709, steps_threshold=[0.6, 0.7, 0.7]):
        """
        Initialize the MTCNN face detector.
        
        Args:
            min_face_size (int): Minimum face size to detect (not used in this MTCNN version)
            scale_factor (float): Scale factor for image pyramid (not used in this MTCNN version)
            steps_threshold (list): Thresholds for the three stages (not used in this MTCNN version)
        """
        try:
            # The mtcnn library we're using has simpler initialization
            self.detector = MTCNN()
            logger.info("MTCNN face detector initialized successfully")
            
            # Store parameters for reference (though not used by this MTCNN version)
            self.min_face_size = min_face_size
            self.scale_factor = scale_factor
            self.steps_threshold = steps_threshold
            
        except Exception as e:
            logger.error(f"Failed to initialize MTCNN: {str(e)}")
            raise
    
    def detect_faces(self, image):
        """
        Detect faces in an image using MTCNN.
        
        Args:
            image (numpy.ndarray): Input image in BGR format (OpenCV format)
            
        Returns:
            list: List of detected faces with bounding boxes, confidence scores, and landmarks
        """
        try:
            # Convert BGR to RGB (MTCNN expects RGB)
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Detect faces
            results = self.detector.detect_faces(rgb_image)
            
            detected_faces = []
            
            for result in results:
                # Extract bounding box
                bbox = result['box']  # [x, y, width, height]
                confidence = result['confidence']
                
                # Extract facial landmarks
                keypoints = result['keypoints']
                
                # Ensure bounding box coordinates are valid
                x, y, w, h = bbox
                x = max(0, x)
                y = max(0, y)
                w = max(1, w)
                h = max(1, h)
                
                # Make sure the bounding box doesn't exceed image dimensions
                img_height, img_width = image.shape[:2]
                x = min(x, img_width - 1)
                y = min(y, img_height - 1)
                w = min(w, img_width - x)
                h = min(h, img_height - y)
                
                face_data = {
                    'bbox': [x, y, w, h],  # [x, y, width, height]
                    'confidence': float(confidence),
                    'landmarks': {
                        'left_eye': keypoints['left_eye'],
                        'right_eye': keypoints['right_eye'],
                        'nose': keypoints['nose'],
                        'mouth_left': keypoints['mouth_left'],
                        'mouth_right': keypoints['mouth_right']
                    }
                }
                
                detected_faces.append(face_data)
            
            logger.info(f"Detected {len(detected_faces)} faces in the image")
            return detected_faces
            
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return []
    
    def extract_face(self, image, bbox, margin=0.2):
        """
        Extract a face region from an image with optional margin.
        
        Args:
            image (numpy.ndarray): Input image
            bbox (list): Bounding box [x, y, width, height]
            margin (float): Margin to add around the face (as fraction of face size)
            
        Returns:
            numpy.ndarray: Extracted face image
        """
        try:
            x, y, w, h = bbox
            img_height, img_width = image.shape[:2]
            
            # Calculate margin
            margin_x = int(w * margin)
            margin_y = int(h * margin)
            
            # Calculate expanded coordinates
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(img_width, x + w + margin_x)
            y2 = min(img_height, y + h + margin_y)
            
            # Extract face region
            face_image = image[y1:y2, x1:x2]
            
            return face_image
            
        except Exception as e:
            logger.error(f"Face extraction failed: {str(e)}")
            return None
    
    def draw_detections(self, image, detections, draw_landmarks=True):
        """
        Draw face detection results on an image.
        
        Args:
            image (numpy.ndarray): Input image
            detections (list): List of face detections
            draw_landmarks (bool): Whether to draw facial landmarks
            
        Returns:
            numpy.ndarray: Image with drawn detections
        """
        try:
            result_image = image.copy()
            
            for detection in detections:
                bbox = detection['bbox']
                confidence = detection['confidence']
                landmarks = detection['landmarks']
                
                x, y, w, h = bbox
                
                # Draw bounding box
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw confidence score
                confidence_text = f"{confidence:.2f}"
                cv2.putText(result_image, confidence_text, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Draw facial landmarks
                if draw_landmarks:
                    for landmark_name, (lx, ly) in landmarks.items():
                        cv2.circle(result_image, (int(lx), int(ly)), 2, (255, 0, 0), -1)
            
            return result_image
            
        except Exception as e:
            logger.error(f"Drawing detections failed: {str(e)}")
            return image
    
    def get_largest_face(self, detections):
        """
        Get the largest face from a list of detections.
        
        Args:
            detections (list): List of face detections
            
        Returns:
            dict: Detection data for the largest face, or None if no faces
        """
        if not detections:
            return None
        
        largest_face = max(detections, key=lambda x: x['bbox'][2] * x['bbox'][3])
        return largest_face
    
    def filter_by_confidence(self, detections, min_confidence=0.9):
        """
        Filter detections by minimum confidence threshold.
        
        Args:
            detections (list): List of face detections
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            list: Filtered detections
        """
        filtered = [det for det in detections if det['confidence'] >= min_confidence]
        logger.info(f"Filtered {len(filtered)} faces with confidence >= {min_confidence}")
        return filtered
