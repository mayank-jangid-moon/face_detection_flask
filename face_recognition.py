import cv2
import numpy as np
import pickle
import os
import json
import sys
from datetime import datetime
import logging
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import math
from face_alignment import FaceAligner

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
    logger.info("Using tflite_runtime")
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        TFLITE_AVAILABLE = True
        logger.info("Using TensorFlow Lite from TensorFlow package")
    except ImportError:
        TFLITE_AVAILABLE = False
        logger.warning("TensorFlow Lite not available. Using fallback approach.")

class MobileFaceNetTFLite:
    """
    MobileFaceNet implementation using TensorFlow Lite.
    This uses the pre-trained model from the Flutter repository.
    """
    
    def __init__(self, model_path='models/mobilefacenet.tflite'):
        """
        Initialize the MobileFaceNet model.
        
        Args:
            model_path (str): Path to the TFLite model file
        """
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_shape = None
        self.output_shape = None
        
        # Initialize face aligner for preprocessing
        self.face_aligner = FaceAligner(target_size=(112, 112), margin_ratio=0.3)
        
        if TFLITE_AVAILABLE:
            self.load_model()
        else:
            logger.error("TensorFlow Lite not available")
    
    def load_model(self):
        """Load the TensorFlow Lite model."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load TFLite model
            if 'tflite_runtime' in sys.modules:
                self.interpreter = tflite.Interpreter(model_path=self.model_path)
            else:
                self.interpreter = tflite.Interpreter(model_path=self.model_path)
            
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.input_shape = self.input_details[0]['shape']
            self.output_shape = self.output_details[0]['shape']
            
            logger.info(f"Model loaded successfully")
            logger.info(f"Input shape: {self.input_shape}")
            logger.info(f"Output shape: {self.output_shape}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.interpreter = None
    
    def preprocess_image(self, image, landmarks=None, bbox=None):
        """
        Preprocess image for MobileFaceNet inference with face alignment.
        
        Args:
            image (numpy.ndarray): Input face image
            landmarks (dict): Facial landmarks for alignment (optional)
            bbox (list): Face bounding box [x, y, w, h] (optional)
            
        Returns:
            numpy.ndarray: Preprocessed image ready for inference
        """
        try:
            # If landmarks are available, use face alignment
            if landmarks is not None:
                aligned_face = self.face_aligner.align_face(image, landmarks, bbox)
            else:
                # Fallback to simple square cropping and resizing
                target_size = (112, 112)
                if bbox is not None:
                    # Extract face region and make it square
                    x, y, w, h = bbox
                    # Add margin
                    margin = int(max(w, h) * 0.3)
                    size = max(w, h) + 2 * margin
                    
                    # Center the square crop
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    x1 = max(0, center_x - size // 2)
                    y1 = max(0, center_y - size // 2)
                    x2 = min(image.shape[1], x1 + size)
                    y2 = min(image.shape[0], y1 + size)
                    
                    # Adjust if we hit boundaries
                    if x2 - x1 < size:
                        x1 = max(0, x2 - size)
                    if y2 - y1 < size:
                        y1 = max(0, y2 - size)
                    
                    face_region = image[y1:y2, x1:x2]
                    
                    # Make square if needed
                    if face_region.shape[0] != face_region.shape[1]:
                        face_region = self.face_aligner._make_square(face_region, max(face_region.shape[:2]))
                    
                    aligned_face = cv2.resize(face_region, target_size, interpolation=cv2.INTER_AREA)
                else:
                    # Just resize the entire image
                    aligned_face = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            
            # Convert BGR to RGB
            if len(aligned_face.shape) == 3 and aligned_face.shape[2] == 3:
                aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values to [-1, 1] as per Flutter implementation
            # Flutter uses: (pixel - mean) / std where mean=128, std=128
            aligned_face = aligned_face.astype(np.float32)
            aligned_face = (aligned_face - 128.0) / 128.0
            
            # Add batch dimension
            aligned_face = np.expand_dims(aligned_face, axis=0)
            
            return aligned_face
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            return None
    
    def extract_embedding(self, face_image, landmarks=None, bbox=None):
        """
        Extract face embedding using MobileFaceNet with face alignment.
        
        Args:
            face_image (numpy.ndarray): Face image
            landmarks (dict): Facial landmarks for alignment (optional)
            bbox (list): Face bounding box (optional)
            
        Returns:
            numpy.ndarray: Face embedding vector (192-dimensional)
        """
        try:
            if self.interpreter is None:
                logger.error("Model not loaded")
                return None
            
            # Preprocess image with alignment
            preprocessed = self.preprocess_image(face_image, landmarks, bbox)
            if preprocessed is None:
                return None
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            embedding = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Flatten the output
            embedding = embedding.flatten()
            
            # Normalize the embedding (L2 normalization)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding extraction failed: {str(e)}")
            return None

class FaceRecognizer:
    """
    Face recognition class using MobileFaceNet TFLite model for generating embeddings
    and performing similarity search against a database of known faces.
    """
    
    def __init__(self, similarity_threshold=1.0, database_path='face_database.pkl', model_path='models/mobilefacenet.tflite'):
        """
        Initialize the face recognizer.
        
        Args:
            similarity_threshold (float): Threshold for face matching (using euclidean distance)
            database_path (str): Path to save/load the face database
            model_path (str): Path to the MobileFaceNet TFLite model
        """
        self.similarity_threshold = similarity_threshold
        self.database_path = database_path
        self.model_path = model_path
        
        # Initialize the model
        if TFLITE_AVAILABLE:
            self.model = MobileFaceNetTFLite(model_path)
        else:
            # Fallback to a simpler approach using traditional computer vision
            self.model = None
            logger.warning("Using fallback face recognition without deep learning")
        
        # Load existing database
        self.face_database = self.load_database()
        
        logger.info(f"Face recognizer initialized with {len(self.face_database)} known faces")
        logger.info(f"Similarity threshold: {self.similarity_threshold}")
    
    def extract_embedding(self, face_image, landmarks=None, bbox=None):
        """
        Extract face embedding using MobileFaceNet TFLite model with face alignment.
        
        Args:
            face_image (numpy.ndarray): Face image
            landmarks (dict): Facial landmarks for alignment (optional)
            bbox (list): Face bounding box (optional)
            
        Returns:
            numpy.ndarray: Face embedding vector
        """
        try:
            if TFLITE_AVAILABLE and self.model and self.model.interpreter:
                # Use the TFLite model with alignment
                return self.model.extract_embedding(face_image, landmarks, bbox)
            else:
                # Fallback: use simple histogram-based features
                return self.extract_simple_features(face_image)
            
        except Exception as e:
            logger.error(f"Embedding extraction failed: {str(e)}")
            return None
    
    def extract_simple_features(self, face_image):
        """
        Fallback method to extract simple features from face image.
        Uses histogram and texture features.
        
        Args:
            face_image (numpy.ndarray): Face image
            
        Returns:
            numpy.ndarray: Feature vector
        """
        try:
            # Resize to standard size
            face_resized = cv2.resize(face_image, (112, 112))
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # Calculate histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-7)  # Normalize
            
            # Calculate LBP (Local Binary Patterns) features
            lbp = self.calculate_lbp(gray)
            lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
            lbp_hist = lbp_hist.flatten()
            lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-7)  # Normalize
            
            # Combine features
            features = np.concatenate([hist, lbp_hist])
            
            return features
            
        except Exception as e:
            logger.error(f"Simple feature extraction failed: {str(e)}")
            return None
    
    def calculate_lbp(self, image):
        """
        Calculate Local Binary Pattern features.
        
        Args:
            image (numpy.ndarray): Grayscale image
            
        Returns:
            numpy.ndarray: LBP image
        """
        try:
            height, width = image.shape
            lbp = np.zeros((height, width), dtype=np.uint8)
            
            for i in range(1, height - 1):
                for j in range(1, width - 1):
                    center = image[i, j]
                    code = 0
                    
                    # Compare with 8 neighbors
                    neighbors = [
                        image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                        image[i, j+1], image[i+1, j+1], image[i+1, j],
                        image[i+1, j-1], image[i, j-1]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            code |= (1 << k)
                    
                    lbp[i, j] = code
            
            return lbp
            
        except Exception as e:
            logger.error(f"LBP calculation failed: {str(e)}")
            return image
    
    def calculate_similarity(self, embedding1, embedding2):
        """
        Calculate similarity between two face embeddings using euclidean distance.
        Lower distance means higher similarity.
        
        Args:
            embedding1 (numpy.ndarray): First embedding
            embedding2 (numpy.ndarray): Second embedding
            
        Returns:
            float: Euclidean distance (lower is more similar)
        """
        try:
            # Calculate euclidean distance (as used in Flutter implementation)
            distance = self.euclidean_distance(embedding1, embedding2)
            return float(distance)
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {str(e)}")
            return float('inf')  # Return large distance on error
    
    def euclidean_distance(self, e1, e2):
        """
        Calculate euclidean distance between two embeddings.
        This matches the implementation in the Flutter app.
        
        Args:
            e1 (numpy.ndarray): First embedding
            e2 (numpy.ndarray): Second embedding
            
        Returns:
            float: Euclidean distance
        """
        try:
            # Ensure arrays are numpy arrays
            e1 = np.array(e1)
            e2 = np.array(e2)
            
            # Calculate euclidean distance
            distance = 0.0
            for i in range(len(e1)):
                distance += (e1[i] - e2[i]) ** 2
            
            return math.sqrt(distance)
            
        except Exception as e:
            logger.error(f"Euclidean distance calculation failed: {str(e)}")
            return float('inf')
    
    def recognize_face(self, face_image, landmarks=None, bbox=None):
        """
        Recognize a face by comparing against the database using euclidean distance.
        
        Args:
            face_image (numpy.ndarray): Face image to recognize
            landmarks (dict): Facial landmarks for alignment (optional)
            bbox (list): Face bounding box (optional)
            
        Returns:
            dict: Recognition result with person info and confidence
        """
        try:
            # Extract embedding from input face with alignment
            embedding = self.extract_embedding(face_image, landmarks, bbox)
            if embedding is None:
                return {
                    'recognized': False,
                    'person_name': 'Unknown',
                    'person_id': None,
                    'confidence': float('inf'),
                    'error': 'Failed to extract embedding'
                }
            
            if not self.face_database:
                return {
                    'recognized': False,
                    'person_name': 'Unknown',
                    'person_id': None,
                    'confidence': float('inf'),
                    'message': 'No faces in database'
                }
            
            # Find best match in database (minimum distance)
            best_match = None
            min_distance = float('inf')
            
            for person_id, person_data in self.face_database.items():
                for face_data in person_data['faces']:
                    distance = self.calculate_similarity(embedding, face_data['embedding'])
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_match = person_data
            
            # Check if distance is below threshold (closer matches have lower distance)
            if min_distance <= self.similarity_threshold:
                return {
                    'recognized': True,
                    'person_name': best_match['name'],
                    'person_id': best_match['id'],
                    'confidence': min_distance,
                    'match_threshold': self.similarity_threshold,
                    'distance': min_distance
                }
            else:
                return {
                    'recognized': False,
                    'person_name': 'Unknown',
                    'person_id': None,
                    'confidence': min_distance,
                    'match_threshold': self.similarity_threshold,
                    'best_match_name': best_match['name'] if best_match else None,
                    'distance': min_distance
                }
            
        except Exception as e:
            logger.error(f"Face recognition failed: {str(e)}")
            return {
                'recognized': False,
                'person_name': 'Unknown',
                'person_id': None,
                'confidence': float('inf'),
                'error': str(e)
            }
    
    def add_person(self, face_image, person_name, person_id=None, landmarks=None, bbox=None):
        """
        Add a new person to the face database.
        
        Args:
            face_image (numpy.ndarray): Face image
            person_name (str): Person's name
            person_id (str): Unique person ID (optional)
            landmarks (dict): Facial landmarks for alignment (optional)
            bbox (list): Face bounding box (optional)
            
        Returns:
            dict: Result of adding person
        """
        try:
            # Generate unique ID if not provided
            if person_id is None:
                person_id = str(uuid.uuid4())
            
            # Extract embedding with alignment
            embedding = self.extract_embedding(face_image, landmarks, bbox)
            if embedding is None:
                return {
                    'success': False,
                    'error': 'Failed to extract face embedding'
                }
            
            # Create face data
            face_data = {
                'embedding': embedding,
                'added_date': datetime.now().isoformat()
            }
            
            # Add to database
            if person_id in self.face_database:
                # Add another face for existing person
                self.face_database[person_id]['faces'].append(face_data)
                self.face_database[person_id]['updated_date'] = datetime.now().isoformat()
            else:
                # Create new person entry
                self.face_database[person_id] = {
                    'id': person_id,
                    'name': person_name,
                    'faces': [face_data],
                    'created_date': datetime.now().isoformat(),
                    'updated_date': datetime.now().isoformat()
                }
            
            # Save database
            self.save_database()
            
            logger.info(f"Added face for person: {person_name} (ID: {person_id})")
            
            return {
                'success': True,
                'person_id': person_id,
                'total_faces': len(self.face_database[person_id]['faces'])
            }
            
        except Exception as e:
            logger.error(f"Failed to add person: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def list_persons(self):
        """
        List all persons in the database.
        
        Returns:
            list: List of person information
        """
        try:
            persons = []
            for person_id, person_data in self.face_database.items():
                persons.append({
                    'id': person_id,
                    'name': person_data['name'],
                    'total_faces': len(person_data['faces']),
                    'created_date': person_data.get('created_date'),
                    'updated_date': person_data.get('updated_date')
                })
            
            return persons
            
        except Exception as e:
            logger.error(f"Failed to list persons: {str(e)}")
            return []
    
    def save_database(self):
        """Save the face database to disk."""
        try:
            with open(self.database_path, 'wb') as f:
                pickle.dump(self.face_database, f)
            logger.info(f"Database saved to {self.database_path}")
        except Exception as e:
            logger.error(f"Failed to save database: {str(e)}")
    
    def load_database(self):
        """Load the face database from disk."""
        try:
            if os.path.exists(self.database_path):
                with open(self.database_path, 'rb') as f:
                    database = pickle.load(f)
                logger.info(f"Database loaded from {self.database_path}")
                return database
            else:
                logger.info("No existing database found, starting fresh")
                return {}
        except Exception as e:
            logger.error(f"Failed to load database: {str(e)}")
            return {}
    
    def delete_person(self, person_id):
        """
        Delete a person from the database.
        
        Args:
            person_id (str): Person ID to delete
            
        Returns:
            dict: Result of deletion
        """
        try:
            if person_id in self.face_database:
                person_name = self.face_database[person_id]['name']
                del self.face_database[person_id]
                self.save_database()
                
                logger.info(f"Deleted person: {person_name} (ID: {person_id})")
                return {
                    'success': True,
                    'message': f'Person {person_name} deleted successfully'
                }
            else:
                return {
                    'success': False,
                    'error': 'Person not found in database'
                }
                
        except Exception as e:
            logger.error(f"Failed to delete person: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
