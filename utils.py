import os
import cv2
import numpy as np
import base64
import logging
from datetime import datetime
from werkzeug.utils import secure_filename

# Set up logging
logger = logging.getLogger(__name__)

def allowed_file(filename, allowed_extensions):
    """
    Check if the uploaded file has an allowed extension.
    
    Args:
        filename (str): Name of the file
        allowed_extensions (set): Set of allowed file extensions
        
    Returns:
        bool: True if file extension is allowed
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def encode_image_to_base64(image_path):
    """
    Encode image to base64 string.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded image string
    """
    try:
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        logger.error(f"Failed to encode image to base64: {str(e)}")
        return None

def decode_base64_to_image(base64_string, output_path):
    """
    Decode base64 string to image file.
    
    Args:
        base64_string (str): Base64 encoded image string
        output_path (str): Path to save the decoded image
        
    Returns:
        bool: True if successful
    """
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        
        with open(output_path, 'wb') as f:
            f.write(image_data)
        
        return True
    except Exception as e:
        logger.error(f"Failed to decode base64 to image: {str(e)}")
        return False

def generate_unique_filename(original_filename, prefix=""):
    """
    Generate a unique filename with timestamp.
    
    Args:
        original_filename (str): Original filename
        prefix (str): Optional prefix for the filename
        
    Returns:
        str: Unique filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Remove last 3 digits from microseconds
    secure_name = secure_filename(original_filename)
    
    if prefix:
        return f"{prefix}_{timestamp}_{secure_name}"
    else:
        return f"{timestamp}_{secure_name}"

def validate_image(image_path):
    """
    Validate if the file is a valid image.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: (is_valid, image_shape, error_message)
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return False, None, "Invalid image file or corrupted image"
        
        return True, image.shape, None
    except Exception as e:
        return False, None, str(e)

def resize_image(image, max_width=1024, max_height=1024):
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image (numpy.ndarray): Input image
        max_width (int): Maximum width
        max_height (int): Maximum height
        
    Returns:
        numpy.ndarray: Resized image
    """
    try:
        height, width = image.shape[:2]
        
        # Calculate scaling factor
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return image
    except Exception as e:
        logger.error(f"Failed to resize image: {str(e)}")
        return image

def create_response_json(status, data=None, error=None, message=None):
    """
    Create standardized JSON response.
    
    Args:
        status (str): Response status ('success' or 'error')
        data (dict): Response data
        error (str): Error message
        message (str): Additional message
        
    Returns:
        dict: Standardized response
    """
    response = {
        'status': status,
        'timestamp': datetime.now().isoformat()
    }
    
    if data is not None:
        response.update(data)
    
    if error is not None:
        response['error'] = error
    
    if message is not None:
        response['message'] = message
    
    return response

def cleanup_file(file_path, delay=None):
    """
    Clean up (delete) a file with optional delay.
    
    Args:
        file_path (str): Path to the file to delete
        delay (float): Optional delay in seconds before deletion
        
    Returns:
        bool: True if successful
    """
    try:
        if delay:
            import time
            time.sleep(delay)
        
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Cleaned up file: {file_path}")
            return True
        
        return True  # File doesn't exist, consider it cleaned up
    except Exception as e:
        logger.error(f"Failed to cleanup file {file_path}: {str(e)}")
        return False

def ensure_directory_exists(directory_path):
    """
    Ensure that a directory exists, create it if it doesn't.
    
    Args:
        directory_path (str): Path to the directory
        
    Returns:
        bool: True if directory exists or was created successfully
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory_path}: {str(e)}")
        return False

def get_image_info(image):
    """
    Get basic information about an image.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        dict: Image information
    """
    try:
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        
        return {
            'width': int(width),
            'height': int(height),
            'channels': int(channels),
            'size': int(image.size),
            'dtype': str(image.dtype)
        }
    except Exception as e:
        logger.error(f"Failed to get image info: {str(e)}")
        return {}

def convert_numpy_types(obj):
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object that might contain numpy types
        
    Returns:
        Object with numpy types converted to Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def log_processing_time(func):
    """
    Decorator to log processing time of functions.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        logger.info(f"{func.__name__} completed in {processing_time:.3f} seconds")
        return result
    
    return wrapper

def validate_confidence_threshold(threshold):
    """
    Validate confidence threshold value.
    
    Args:
        threshold (float): Threshold value to validate
        
    Returns:
        tuple: (is_valid, validated_threshold, error_message)
    """
    try:
        threshold = float(threshold)
        if 0.0 <= threshold <= 1.0:
            return True, threshold, None
        else:
            return False, None, "Threshold must be between 0.0 and 1.0"
    except (ValueError, TypeError):
        return False, None, "Threshold must be a valid number"
