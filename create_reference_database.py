#!/usr/bin/env python3
"""
Reference Image Database Generator
Processes all images in the images folder to create a reference database
with embeddings for face recognition.
"""

import os
import json
import cv2
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import glob

# Import our modules
from face_detection import FaceDetector
from face_recognition import FaceRecognizer
from face_alignment import FaceAligner

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

class ReferenceImageProcessor:
    """
    Process reference images to create a face recognition database.
    """
    
    def __init__(self, images_folder="images", output_folder=None):
        """
        Initialize the reference image processor.
        
        Args:
            images_folder (str): Path to folder containing reference images
            output_folder (str): Path to save outputs (auto-generated if None)
        """
        self.images_folder = images_folder
        self.output_folder = output_folder or f"reference_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize components
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.face_aligner = FaceAligner()
        
        # Create output directories
        self.setup_output_directories()
        
        # Reference database
        self.reference_database = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_images": 0,
                "successful_extractions": 0,
                "failed_extractions": 0,
                "preprocessing_method": "aligned_face_with_centering"
            },
            "people": {}
        }
    
    def setup_output_directories(self):
        """Create all necessary output directories."""
        base_path = Path(self.output_folder)
        
        self.dirs = {
            "base": base_path,
            "processed": base_path / "processed_images",
            "embeddings": base_path / "embeddings",
            "debug": base_path / "debug_steps",
            "logs": base_path / "logs",
            "failed": base_path / "failed_processing"
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directories created in: {self.output_folder}")
    
    def get_person_name_from_filename(self, filename):
        """
        Extract person name from filename.
        
        Args:
            filename (str): Image filename
            
        Returns:
            str: Person name (cleaned)
        """
        # Remove file extension and clean up
        name = Path(filename).stem
        
        # Clean up common filename patterns
        name = name.replace("_", " ").replace("-", " ")
        name = " ".join(word.capitalize() for word in name.split())
        
        return name
    
    def process_single_image(self, image_path):
        """
        Process a single reference image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Processing results
        """
        filename = os.path.basename(image_path)
        person_name = self.get_person_name_from_filename(filename)
        
        logger.info(f"Processing {filename} for person: {person_name}")
        
        result = {
            "filename": filename,
            "person_name": person_name,
            "success": False,
            "faces_detected": 0,
            "embedding": None,
            "face_bbox": None,
            "landmarks": None,
            "error": None
        }
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            result["image_shape"] = list(image.shape)
            
            # Create debug directory for this person
            person_debug_dir = self.dirs["debug"] / person_name.replace(" ", "_")
            person_debug_dir.mkdir(exist_ok=True)
            
            # Save original image
            cv2.imwrite(str(person_debug_dir / "01_original.jpg"), image)
            
            # Detect faces
            faces = self.face_detector.detect_faces(image)
            result["faces_detected"] = len(faces)
            
            if len(faces) == 0:
                raise ValueError("No faces detected in image")
            
            if len(faces) > 1:
                logger.warning(f"Multiple faces detected in {filename}, using the largest one")
            
            # Use the face with highest confidence (first one after sorting)
            face = faces[0]
            bbox = face['bbox']
            landmarks = face['landmarks']
            confidence = face['confidence']
            
            # Convert numpy types to native Python types for JSON serialization
            result["face_bbox"] = convert_numpy_types(bbox)
            result["landmarks"] = convert_numpy_types(landmarks)
            result["detection_confidence"] = convert_numpy_types(confidence)
            
            # Draw detection on image for debugging
            debug_image = image.copy()
            x, y, w, h = [int(x) for x in bbox]  # Convert to int for drawing
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw landmarks
            for landmark_name, (lx, ly) in landmarks.items():
                cv2.circle(debug_image, (int(lx), int(ly)), 3, (0, 0, 255), -1)
                cv2.putText(debug_image, landmark_name[:3], (int(lx) + 5, int(ly)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            cv2.imwrite(str(person_debug_dir / "02_detected_face.jpg"), debug_image)
            
            # Process face with alignment and centering
            aligned_face, debug_info = self.face_aligner.preprocess_face_with_debug(
                image, landmarks, bbox, save_steps=True, output_dir=str(person_debug_dir)
            )
            
            # Save debug info (convert numpy types)
            with open(person_debug_dir / "preprocessing_debug.json", 'w') as f:
                json.dump(convert_numpy_types(debug_info), f, indent=2)
            
            # Extract embedding
            embedding = self.face_recognizer.extract_embedding(aligned_face)
            
            if embedding is not None:
                result["embedding"] = embedding.tolist()
                result["embedding_norm"] = float(np.linalg.norm(embedding))
                result["embedding_shape"] = list(embedding.shape)
                result["success"] = True
                
                # Save embedding as numpy array
                embedding_file = self.dirs["embeddings"] / f"{person_name.replace(' ', '_')}_embedding.npy"
                np.save(embedding_file, embedding)
                
                # Save final processed face
                final_face_file = self.dirs["processed"] / f"{person_name.replace(' ', '_')}_processed.jpg"
                cv2.imwrite(str(final_face_file), aligned_face)
                
                logger.info(f"✅ Successfully processed {person_name}")
            else:
                raise ValueError("Failed to extract embedding")
                
        except Exception as e:
            error_msg = str(e)
            result["error"] = error_msg
            result["success"] = False
            logger.error(f"❌ Failed to process {filename}: {error_msg}")
            
            # Move failed image to failed directory
            failed_file = self.dirs["failed"] / filename
            try:
                import shutil
                shutil.copy2(image_path, failed_file)
            except Exception as copy_error:
                logger.error(f"Could not copy failed file: {copy_error}")
        
        return result
    
    def process_all_images(self):
        """
        Process all images in the images folder.
        
        Returns:
            dict: Complete reference database
        """
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        
        for ext in image_extensions:
            pattern = os.path.join(self.images_folder, ext)
            image_files.extend(glob.glob(pattern))
        
        logger.info(f"Found {len(image_files)} image files to process")
        self.reference_database["metadata"]["total_images"] = len(image_files)
        
        # Process each image
        for image_path in sorted(image_files):
            result = self.process_single_image(image_path)
            
            if result["success"]:
                person_name = result["person_name"]
                
                # Store in reference database
                self.reference_database["people"][person_name] = {
                    "filename": result["filename"],
                    "embedding": result["embedding"],
                    "embedding_norm": result["embedding_norm"],
                    "embedding_shape": result["embedding_shape"],
                    "face_bbox": result["face_bbox"],
                    "landmarks": result["landmarks"],
                    "detection_confidence": result["detection_confidence"],
                    "image_shape": result["image_shape"],
                    "processed_at": datetime.now().isoformat()
                }
                
                self.reference_database["metadata"]["successful_extractions"] += 1
            else:
                self.reference_database["metadata"]["failed_extractions"] += 1
            
            # Save individual result log
            result_file = self.dirs["logs"] / f"{result['person_name'].replace(' ', '_')}_result.json"
            with open(result_file, 'w') as f:
                json.dump(convert_numpy_types(result), f, indent=2)
        
        # Save complete reference database
        database_file = self.dirs["base"] / "reference_database.json"
        with open(database_file, 'w') as f:
            json.dump(convert_numpy_types(self.reference_database), f, indent=2)
        
        # Create summary report
        self.create_summary_report()
        
        return self.reference_database
    
    def create_summary_report(self):
        """Create a summary report of the processing."""
        metadata = self.reference_database["metadata"]
        people = self.reference_database["people"]
        
        report = f"""
REFERENCE DATABASE GENERATION REPORT
=====================================

Processing Summary:
- Total images: {metadata['total_images']}
- Successfully processed: {metadata['successful_extractions']}
- Failed processing: {metadata['failed_extractions']}
- Success rate: {(metadata['successful_extractions'] / metadata['total_images'] * 100):.1f}%

People in Database:
"""
        
        for i, (name, data) in enumerate(sorted(people.items()), 1):
            report += f"{i:2d}. {name:<20} | Embedding: {data['embedding_shape']} | Confidence: {data['detection_confidence']:.3f}\n"
        
        report += f"""
Database Statistics:
- Average embedding norm: {np.mean([data['embedding_norm'] for data in people.values()]):.6f}
- Embedding dimension: {list(people.values())[0]['embedding_shape'] if people else 'N/A'}

Files Generated:
- Reference database: reference_database.json
- Individual embeddings: embeddings/*.npy
- Processed faces: processed_images/*.jpg
- Debug steps: debug_steps/*/
- Processing logs: logs/*.json

Database ready for face recognition!
"""
        
        # Save report
        report_file = self.dirs["base"] / "PROCESSING_REPORT.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Print to console
        print(report)
        
        logger.info(f"Reference database created successfully in: {self.output_folder}")

def main():
    """Main function to process reference images."""
    print("🚀 Starting Reference Image Database Generation...")
    
    # Check if images folder exists
    if not os.path.exists("images"):
        print("❌ Images folder not found!")
        return
    
    # Create processor and run
    processor = ReferenceImageProcessor()
    database = processor.process_all_images()
    
    print(f"✅ Reference database generation completed!")
    print(f"📁 Results saved in: {processor.output_folder}")

if __name__ == "__main__":
    main()
