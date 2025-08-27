#!/usr/bin/env python3
"""
Enhanced Face Recognition Test Script
Processes an image, detects faces, performs similarity search with reference database,
and annotates the image with identified people.
"""

import cv2
import numpy as np
import os
import json
import sys
from datetime import datetime
import logging
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from face_detection import FaceDetector
from face_recognition import FaceRecognizer
from face_alignment import FaceAligner

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceRecognitionTester:
    """
    Test face recognition with reference database and image annotation.
    """
    
    def __init__(self, reference_db_path="reference_database.json", similarity_threshold=0.7):
        """
        Initialize the face recognition tester.
        
        Args:
            reference_db_path (str): Path to reference database JSON file
            similarity_threshold (float): Threshold for face recognition (higher = more strict for cosine similarity)
        """
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.face_aligner = FaceAligner()
        self.similarity_threshold = similarity_threshold
        
        # Load reference database
        self.reference_database = self.load_reference_database(reference_db_path)
        logger.info(f"Loaded reference database with {len(self.reference_database['people'])} people")
        
        # Create output directory
        self.output_dir = self.create_output_directory()
    
    def load_reference_database(self, db_path):
        """Load the reference database from JSON file."""
        try:
            with open(db_path, 'r') as f:
                database = json.load(f)
            logger.info(f"Successfully loaded reference database: {db_path}")
            return database
        except FileNotFoundError:
            logger.error(f"Reference database not found: {db_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing reference database: {e}")
            raise
    
    def create_output_directory(self):
        """Create output directory for test results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"face_recognition_test_{timestamp}"
        Path(output_dir).mkdir(exist_ok=True)
        
        # Create subdirectories
        subdirs = ['annotated_images', 'detected_faces', 'debug_steps', 'results']
        for subdir in subdirs:
            Path(output_dir, subdir).mkdir(exist_ok=True)
        
        logger.info(f"Created output directory: {output_dir}")
        return output_dir
    
    def calculate_similarity(self, embedding1, embedding2):
        """
        Calculate similarity between two embeddings using cosine similarity.
        
        Args:
            embedding1 (list): First embedding
            embedding2 (list): Second embedding
            
        Returns:
            float: Similarity score (higher = more similar, range: -1 to 1)
        """
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return -1.0  # Maximum dissimilarity
        
        cosine_sim = dot_product / (norm1 * norm2)
        return cosine_sim
    
    def find_best_match(self, face_embedding):
        """
        Find the best matching person in the reference database.
        
        Args:
            face_embedding (numpy.ndarray): Face embedding to match
            
        Returns:
            tuple: (person_name, similarity_score, confidence_level)
        """
        if face_embedding is None:
            return None, -1.0, "No Embedding"
        
        best_match = None
        best_similarity = -1.0  # Start with lowest possible cosine similarity
        
        face_emb_list = face_embedding.tolist()
        
        for person_name, person_data in self.reference_database['people'].items():
            ref_embedding = person_data['embedding']
            similarity = self.calculate_similarity(face_emb_list, ref_embedding)
            
            if similarity > best_similarity:  # Higher cosine similarity = better match
                best_similarity = similarity
                best_match = person_name
        
        # Determine confidence level (higher cosine similarity = better)
        if best_similarity > self.similarity_threshold:
            confidence = "HIGH"
        elif best_similarity > self.similarity_threshold - 0.15:  # Adjusted range for cosine
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        return best_match, best_similarity, confidence
    
    def draw_face_annotation(self, image, bbox, person_name, similarity, confidence, face_id):
        """
        Draw face annotation on the image.
        
        Args:
            image (numpy.ndarray): Image to annotate
            bbox (list): Bounding box [x, y, w, h]
            person_name (str): Identified person name
            similarity (float): Similarity score
            confidence (str): Confidence level
            face_id (int): Face identifier
            
        Returns:
            numpy.ndarray: Annotated image
        """
        x, y, w, h = [int(coord) for coord in bbox]
        
        # Choose colors based on confidence
        color_map = {
            "HIGH": (0, 255, 0),      # Green
            "MEDIUM": (0, 165, 255),   # Orange
            "LOW": (0, 0, 255),        # Red
            "No Embedding": (128, 128, 128)  # Gray
        }
        
        color = color_map.get(confidence, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # Prepare text
        if person_name:
            name_text = person_name
            similarity_text = f"Sim: {similarity:.3f}"
        else:
            name_text = "Unknown"
            similarity_text = f"Sim: {similarity:.3f}"
        
        face_id_text = f"Face {face_id}"
        
        # Calculate text size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Get text sizes
        (name_w, name_h), _ = cv2.getTextSize(name_text, font, font_scale, thickness)
        (sim_w, sim_h), _ = cv2.getTextSize(similarity_text, font, font_scale, thickness)
        (id_w, id_h), _ = cv2.getTextSize(face_id_text, font, font_scale, thickness)
        
        # Draw background rectangles for text
        text_bg_color = (0, 0, 0)
        text_color = (255, 255, 255)
        
        # Name background
        cv2.rectangle(image, (x, y - name_h - 10), (x + name_w + 10, y), text_bg_color, -1)
        cv2.putText(image, name_text, (x + 5, y - 5), font, font_scale, text_color, thickness)
        
        # Similarity background
        cv2.rectangle(image, (x, y + h), (x + sim_w + 10, y + h + sim_h + 10), text_bg_color, -1)
        cv2.putText(image, similarity_text, (x + 5, y + h + sim_h + 5), font, font_scale, text_color, thickness)
        
        # Face ID background (top-left corner of bbox)
        cv2.rectangle(image, (x - 2, y - id_h - 35), (x + id_w + 8, y - 25), color, -1)
        cv2.putText(image, face_id_text, (x + 3, y - 30), font, font_scale, (255, 255, 255), thickness)
        
        return image
    
    def process_image(self, image_path):
        """
        Process an image and perform face recognition.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Results of face recognition
        """
        logger.info(f"Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        original_image = image.copy()
        
        # Save original image
        cv2.imwrite(os.path.join(self.output_dir, "annotated_images", "01_original.jpg"), original_image)
        
        # Detect faces
        faces = self.face_detector.detect_faces(image)
        logger.info(f"Detected {len(faces)} faces")
        
        if len(faces) == 0:
            logger.warning("No faces detected in the image")
            return {"status": "no_faces", "faces": []}
        
        results = {
            "status": "success",
            "image_path": image_path,
            "total_faces": len(faces),
            "faces": []
        }
        
        annotated_image = original_image.copy()
        
        # Process each face
        for i, face in enumerate(faces):
            face_id = i + 1
            logger.info(f"Processing face {face_id}/{len(faces)}")
            
            bbox = face['bbox']
            landmarks = face['landmarks']
            detection_confidence = face['confidence']
            
            # Create debug directory for this face
            face_debug_dir = Path(self.output_dir, "debug_steps", f"face_{face_id}")
            face_debug_dir.mkdir(exist_ok=True)
            
            # Save detected face region
            x, y, w, h = [int(coord) for coord in bbox]
            face_crop = original_image[y:y+h, x:x+w]
            cv2.imwrite(str(face_debug_dir / "01_detected_face.jpg"), face_crop)
            
            try:
                # Align and preprocess face
                aligned_face, debug_info = self.face_aligner.preprocess_face_with_debug(
                    original_image, landmarks, bbox, save_steps=True, output_dir=str(face_debug_dir)
                )
                
                # Extract embedding
                embedding = self.face_recognizer.extract_embedding(aligned_face)
                
                if embedding is not None:
                    # Find best match
                    person_name, similarity, confidence = self.find_best_match(embedding)
                    
                    # Save individual results
                    face_result = {
                        "face_id": face_id,
                        "bbox": [int(coord) for coord in bbox],
                        "landmarks": {k: [float(v[0]), float(v[1])] for k, v in landmarks.items()},
                        "detection_confidence": float(detection_confidence),
                        "identified_person": person_name,
                        "similarity_score": float(similarity),
                        "confidence_level": confidence,
                        "embedding": embedding.tolist(),
                        "embedding_shape": list(embedding.shape)
                    }
                    
                    results["faces"].append(face_result)
                    
                    # Save debug info
                    with open(face_debug_dir / "recognition_result.json", 'w') as f:
                        json.dump(face_result, f, indent=2)
                    
                    # Annotate image
                    annotated_image = self.draw_face_annotation(
                        annotated_image, bbox, person_name, similarity, confidence, face_id
                    )
                    
                    logger.info(f"Face {face_id}: {person_name} (similarity: {similarity:.3f}, confidence: {confidence})")
                    
                else:
                    logger.error(f"Failed to extract embedding for face {face_id}")
                    face_result = {
                        "face_id": face_id,
                        "bbox": [int(coord) for coord in bbox],
                        "error": "Failed to extract embedding"
                    }
                    results["faces"].append(face_result)
                    
                    # Still annotate with error
                    annotated_image = self.draw_face_annotation(
                        annotated_image, bbox, None, -1.0, "No Embedding", face_id
                    )
                    
            except Exception as e:
                logger.error(f"Error processing face {face_id}: {str(e)}")
                face_result = {
                    "face_id": face_id,
                    "bbox": [int(coord) for coord in bbox],
                    "error": str(e)
                }
                results["faces"].append(face_result)
        
        # Save annotated image
        annotated_path = os.path.join(self.output_dir, "annotated_images", "02_annotated_result.jpg")
        cv2.imwrite(annotated_path, annotated_image)
        logger.info(f"Saved annotated image: {annotated_path}")
        
        # Save complete results
        results_path = os.path.join(self.output_dir, "results", "recognition_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary report
        self.create_summary_report(results)
        
        return results
    
    def create_summary_report(self, results):
        """Create a summary report of face recognition results."""
        report = f"""
FACE RECOGNITION TEST REPORT
============================

Image: {results['image_path']}
Total faces detected: {results['total_faces']}
Processing timestamp: {datetime.now().isoformat()}

Face Recognition Results:
-------------------------
"""
        
        if results['status'] == 'no_faces':
            report += "No faces detected in the image.\n"
        else:
            for face in results['faces']:
                face_id = face['face_id']
                person = face.get('identified_person', 'Unknown')
                similarity = face.get('similarity_score', 'N/A')
                confidence = face.get('confidence_level', 'N/A')
                
                report += f"Face {face_id}: {person}\n"
                report += f"  - Similarity: {similarity}\n"
                report += f"  - Confidence: {confidence}\n"
                report += f"  - Detection confidence: {face.get('detection_confidence', 'N/A')}\n\n"
        
        report += f"""
Recognition Statistics:
----------------------
- Similarity threshold: {self.similarity_threshold} (cosine similarity)
- Reference database: {len(self.reference_database['people'])} people
- High confidence matches: {len([f for f in results.get('faces', []) if f.get('confidence_level') == 'HIGH'])}
- Medium confidence matches: {len([f for f in results.get('faces', []) if f.get('confidence_level') == 'MEDIUM'])}
- Low confidence matches: {len([f for f in results.get('faces', []) if f.get('confidence_level') == 'LOW'])}

Output Files:
-------------
- Annotated image: annotated_images/02_annotated_result.jpg
- Individual face results: debug_steps/face_*/recognition_result.json
- Complete results: results/recognition_results.json
- Debug steps: debug_steps/face_*/

Legend:
-------
🟢 GREEN box: High confidence match (cosine similarity > {self.similarity_threshold})
🟠 ORANGE box: Medium confidence match (cosine similarity > {self.similarity_threshold - 0.15})
🔴 RED box: Low confidence match (cosine similarity <= {self.similarity_threshold - 0.15})
⚫ GRAY box: No embedding extracted
"""
        
        # Save report
        report_path = os.path.join(self.output_dir, "RECOGNITION_REPORT.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(report)
        print(f"\n📁 All results saved in: {self.output_dir}")

def main():
    """Main function to run face recognition test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Face Recognition Test with Reference Database")
    parser.add_argument("image_path", help="Path to the image to process")
    parser.add_argument("--reference-db", default="reference_database.json", 
                       help="Path to reference database JSON file")
    parser.add_argument("--threshold", type=float, default=0.7, 
                       help="Similarity threshold for recognition (cosine similarity, higher = more strict)")
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"❌ Image not found: {args.image_path}")
        return
    
    # Check if reference database exists
    if not os.path.exists(args.reference_db):
        print(f"❌ Reference database not found: {args.reference_db}")
        print("Please run create_reference_database.py first to create the reference database.")
        return
    
    print("🚀 Starting Face Recognition Test...")
    
    try:
        # Create tester and process image
        tester = FaceRecognitionTester(args.reference_db, args.threshold)
        results = tester.process_image(args.image_path)
        
        print("✅ Face recognition test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during face recognition test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
