#!/usr/bin/env python3
"""
Face Recognition API Client
A comprehensive script that calls the Face Recognition API, processes the response,
and saves the annotated image and metadata separately.
"""

import requests
import base64
import json
import cv2
import numpy as np
import os
import sys
from datetime import datetime
from pathlib import Path
import argparse

class FaceRecognitionAPIClient:
    """Client for interacting with the Face Recognition API."""
    
    def __init__(self, api_url="http://localhost:5000/recognize_faces"):
        """
        Initialize the API client.
        
        Args:
            api_url (str): URL of the face recognition API endpoint
        """
        self.api_url = api_url
        self.output_dir = None
        
    def image_to_base64(self, image_path):
        """Convert an image file to base64 string."""
        try:
            with open(image_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
        except Exception as e:
            raise ValueError(f"Error encoding image: {e}")
    
    def base64_to_image(self, base64_string, output_path):
        """Convert base64 string to image file."""
        try:
            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_string)
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # Decode to OpenCV image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is not None:
                # Save the image
                cv2.imwrite(output_path, image)
                return True
            else:
                print("❌ Failed to decode base64 image")
                return False
                
        except Exception as e:
            print(f"❌ Error converting base64 to image: {e}")
            return False
    
    def create_output_directory(self, base_name):
        """Create output directory for results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"api_results_{base_name}_{timestamp}"
        Path(self.output_dir).mkdir(exist_ok=True)
        
        print(f"📁 Created output directory: {self.output_dir}")
        return self.output_dir
    
    def call_api(self, image_path):
        """
        Call the face recognition API with an image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: API response or None if failed
        """
        print(f"🚀 Calling Face Recognition API")
        print(f"📸 Image: {image_path}")
        print(f"🌐 API URL: {self.api_url}")
        print("=" * 60)
        
        # Check if image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Convert image to base64
        try:
            base64_image = self.image_to_base64(image_path)
            print(f"✅ Image encoded to base64 ({len(base64_image)} characters)")
        except Exception as e:
            raise ValueError(f"Failed to encode image: {e}")
        
        # Prepare request data
        request_data = {
            "image_data": base64_image
        }
        
        try:
            print("📤 Sending request to API...")
            
            # Send POST request
            response = requests.post(
                self.api_url,
                json=request_data,
                headers={'Content-Type': 'application/json'},
                timeout=60  # Increased timeout for processing
            )
            
            print(f"📥 Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ API call successful")
                return result
            else:
                print("❌ API Error:")
                try:
                    error_info = response.json()
                    print(json.dumps(error_info, indent=2))
                except:
                    print(response.text)
                return None
        
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None
    
    def process_response(self, response_data, original_image_path):
        """
        Process the API response and save results.
        
        Args:
            response_data (dict): API response data
            original_image_path (str): Path to original image
            
        Returns:
            dict: Processing summary
        """
        if not response_data:
            print("❌ No response data to process")
            return None
        
        # Create output directory
        base_name = Path(original_image_path).stem
        output_dir = self.create_output_directory(base_name)
        
        # Extract data from response
        results_json = response_data.get('results_json', [])
        processed_image_base64 = response_data.get('processed_image_base64')
        
        print(f"\n📊 Processing Results:")
        print(f"Total faces detected: {len(results_json)}")
        
        # Save original image for reference
        original_output_path = os.path.join(output_dir, "01_original_image.jpg")
        original_image = cv2.imread(original_image_path)
        if original_image is not None:
            cv2.imwrite(original_output_path, original_image)
            print(f"📸 Saved original image: {original_output_path}")
        
        # Save annotated image if available
        annotated_image_path = None
        if processed_image_base64:
            annotated_image_path = os.path.join(output_dir, "02_annotated_image.jpg")
            if self.base64_to_image(processed_image_base64, annotated_image_path):
                print(f"🎨 Saved annotated image: {annotated_image_path}")
            else:
                print("❌ Failed to save annotated image")
                annotated_image_path = None
        else:
            print("⚠️  No annotated image in response")
        
        # Process face detection results
        face_details = []
        for i, face in enumerate(results_json, 1):
            bbox = face.get('boundingBox', {})
            name = face.get('name', 'Unknown')
            similarity = face.get('similarityScore', 0.0)
            
            face_info = {
                "face_id": i,
                "person_name": name,
                "similarity_score": similarity,
                "bounding_box": {
                    "left": bbox.get('left', 0),
                    "top": bbox.get('top', 0),
                    "width": bbox.get('width', 0),
                    "height": bbox.get('height', 0)
                },
                "confidence_level": self._get_confidence_level(similarity)
            }
            
            face_details.append(face_info)
            
            print(f"Face {i}: {name} (similarity: {similarity:.3f}, confidence: {face_info['confidence_level']})")
            print(f"  BBox: left={bbox.get('left', 0)}, top={bbox.get('top', 0)}, width={bbox.get('width', 0)}, height={bbox.get('height', 0)}")
        
        # Create comprehensive metadata
        metadata = {
            "api_call_info": {
                "timestamp": datetime.now().isoformat(),
                "api_url": self.api_url,
                "original_image_path": original_image_path,
                "original_image_name": Path(original_image_path).name
            },
            "processing_results": {
                "total_faces_detected": len(results_json),
                "faces": face_details
            },
            "file_info": {
                "output_directory": output_dir,
                "original_image_saved": original_output_path,
                "annotated_image_saved": annotated_image_path,
                "metadata_file": os.path.join(output_dir, "metadata.json")
            },
            "statistics": {
                "high_confidence_matches": len([f for f in face_details if f['confidence_level'] == 'HIGH']),
                "medium_confidence_matches": len([f for f in face_details if f['confidence_level'] == 'MEDIUM']),
                "low_confidence_matches": len([f for f in face_details if f['confidence_level'] == 'LOW']),
                "unknown_faces": len([f for f in face_details if f['person_name'] == 'Unknown'])
            },
            "raw_api_response": response_data
        }
        
        # Save metadata to JSON file
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"📋 Saved metadata: {metadata_path}")
        
        # Create summary report
        self._create_summary_report(metadata, output_dir)
        
        return metadata
    
    def _get_confidence_level(self, similarity_score):
        """Determine confidence level based on similarity score."""
        if similarity_score > 0.7:
            return "HIGH"
        elif similarity_score > 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _create_summary_report(self, metadata, output_dir):
        """Create a human-readable summary report."""
        report_path = os.path.join(output_dir, "SUMMARY_REPORT.txt")
        
        api_info = metadata['api_call_info']
        results = metadata['processing_results']
        stats = metadata['statistics']
        
        report = f"""
FACE RECOGNITION API RESULTS
============================

API Call Information:
--------------------
Timestamp: {api_info['timestamp']}
API URL: {api_info['api_url']}
Original Image: {api_info['original_image_name']}

Recognition Results:
-------------------
Total Faces Detected: {results['total_faces_detected']}

Face Details:
"""
        
        for face in results['faces']:
            report += f"""
Face {face['face_id']}:
  Name: {face['person_name']}
  Similarity Score: {face['similarity_score']:.3f}
  Confidence: {face['confidence_level']}
  Bounding Box: ({face['bounding_box']['left']}, {face['bounding_box']['top']}, {face['bounding_box']['width']}, {face['bounding_box']['height']})
"""
        
        report += f"""
Statistics:
----------
High Confidence Matches: {stats['high_confidence_matches']}
Medium Confidence Matches: {stats['medium_confidence_matches']}
Low Confidence Matches: {stats['low_confidence_matches']}
Unknown Faces: {stats['unknown_faces']}

Output Files:
------------
Original Image: 01_original_image.jpg
Annotated Image: 02_annotated_image.jpg
Metadata: metadata.json
Summary Report: SUMMARY_REPORT.txt

Confidence Levels:
-----------------
HIGH: Similarity > 0.9 (Very likely match)
MEDIUM: Similarity > 0.7 (Probable match)
LOW: Similarity ≤ 0.7 (Uncertain match)
"""
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"📄 Saved summary report: {report_path}")
        
        # Also print summary to console
        print(f"\n" + "=" * 60)
        print("📊 PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total faces: {results['total_faces_detected']}")
        print(f"High confidence: {stats['high_confidence_matches']}")
        print(f"Medium confidence: {stats['medium_confidence_matches']}")
        print(f"Low confidence: {stats['low_confidence_matches']}")
        print(f"Unknown faces: {stats['unknown_faces']}")
        print(f"📁 All results saved in: {output_dir}")
    
    def run(self, image_path):
        """
        Run the complete face recognition pipeline.
        
        Args:
            image_path (str): Path to the image to process
            
        Returns:
            dict: Processing results metadata
        """
        try:
            # Call the API
            response_data = self.call_api(image_path)
            
            if response_data:
                # Process and save results
                metadata = self.process_response(response_data, image_path)
                print("\n✅ Face recognition processing completed successfully!")
                return metadata
            else:
                print("\n❌ Face recognition processing failed!")
                return None
                
        except Exception as e:
            print(f"\n❌ Error during processing: {e}")
            return None

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Face Recognition API Client")
    parser.add_argument("image_path", help="Path to the image to process")
    parser.add_argument("--api-url", default="http://localhost:5000/recognize_faces", 
                       help="Face recognition API endpoint URL")
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"❌ Image not found: {args.image_path}")
        sys.exit(1)
    
    # Create API client and run
    client = FaceRecognitionAPIClient(args.api_url)
    metadata = client.run(args.image_path)
    
    if metadata:
        print(f"\n🎉 Processing complete! Check the output directory for results.")
    else:
        print(f"\n💥 Processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
