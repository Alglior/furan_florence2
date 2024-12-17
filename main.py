import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, ProcessorMixin
import gc
from PIL import Image
import sys
import csv  # Add this import at the top with other imports

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('florence_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Data class for storing image processing results."""
    description: str
    folder: str
    file_name: str
    error: Optional[str] = None

class Florence2Error(Exception):
    """Custom exception for Florence2-specific errors."""
    pass

class CSVWriter:
    """Handler for incremental CSV writing."""
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file = None
        self.writer = None
        self.initialize()
    
    def initialize(self):
        """Create CSV file with headers."""
        self.file = open(self.file_path, 'w', encoding='utf-8', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['Folder', 'Filename', 'Path', 'Description', 'Error'])
    
    def write_row(self, image_path: str, result: ProcessingResult):
        """Write a single result row to CSV."""
        self.writer.writerow([
            result.folder,
            result.file_name,
            image_path,
            result.description,
            result.error or ''
        ])
        self.file.flush()  # Ensure immediate write to disk
    
    def close(self):
        """Close the CSV file."""
        if self.file:
            self.file.close()

def setup_model(model_id: str = 'microsoft/Florence-2-large') -> Tuple[AutoModelForCausalLM, ProcessorMixin]:
    """Initialize the model and processor with enhanced configuration."""
    try:
        logger.info(f"Loading model: {model_id}")
        
        # Determine device and dtype
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        logger.info(f"Using device: {device}, dtype: {dtype}")
        
        # Load model with explicit dtype
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device)
        
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model.eval()
        
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return model, processor
    except Exception as e:
        raise Florence2Error(f"Failed to initialize model: {str(e)}")

def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocess image for better model input."""
    # Ensure RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize if too large while maintaining aspect ratio
    max_size = 1024
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    return image

def get_image_description(
    image_path: Path,
    model: AutoModelForCausalLM,
    processor: ProcessorMixin
) -> str:
    """Generate detailed description for a single image with enhanced processing."""
    try:
        image = Image.open(image_path)
        image = preprocess_image(image)
        
        # Simplified prompt for better results
        prompt = "Describe this image in detail."
        
        logger.debug(f"Processing image {image_path} with prompt: {prompt}")
        
        # Process image and text inputs
        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        )
        
        # Get model's device
        device = next(model.parameters()).device
        
        # Handle tensor types correctly
        processed_inputs = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                # Convert to appropriate dtype based on tensor type
                if v.dtype == torch.int64 or v.dtype == torch.int32:
                    processed_inputs[k] = v.to(device)
                else:
                    processed_inputs[k] = v.to(device, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
                logger.debug(f"Input tensor {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                processed_inputs[k] = v

        with torch.no_grad():
            logger.debug("Generating description...")
            outputs = model.generate(
                **processed_inputs,
                max_new_tokens=300,
                num_beams=5,
                temperature=0.7,  # Slightly reduced for more focused descriptions
                top_k=50,
                top_p=0.9,
                do_sample=True,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                repetition_penalty=1.2  # Added to prevent repetitive text
            )

        # Decode and clean up the text
        raw_text = processor.decode(outputs[0], skip_special_tokens=True)
        logger.debug(f"Raw model output: {raw_text}")
        
        # Remove the prompt and clean up the text
        description = raw_text.replace(prompt, "").strip()
        description = description.replace("This image shows ", "").strip()  # Remove common prefix
        description = description.replace("The image shows ", "").strip()  # Remove common prefix
        description = description.replace("This is ", "").strip()  # Remove common prefix
        
        # Log the final cleaned description
        logger.debug(f"Cleaned description: {description}")
        
        # Clean up memory
        del outputs, inputs, processed_inputs
        torch.cuda.empty_cache()
        
        return description
    except Exception as e:
        logger.error(f"Error in text generation/processing: {str(e)}", exc_info=True)
        raise Florence2Error(f"Error processing {image_path}: {str(e)}")

def should_process_folder(folder_path):
    """Check if folder should be processed based on name criteria."""
    return folder_path.name.endswith('sans_texte')

def process_image_folder(folder_path: Path) -> Dict[str, ProcessingResult]:
    """Process all images in folders ending with 'sans_texte' with enhanced handling."""
    model, processor = setup_model()
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    root_folder = Path(folder_path)
    results: Dict[str, ProcessingResult] = {}

    # Initialize CSV writer
    csv_path = root_folder / "image_descriptions_sans_texte.csv"
    csv_writer = CSVWriter(csv_path)

    try:
        matching_folders = [d for d in root_folder.rglob('*') 
                          if d.is_dir() and should_process_folder(d)]
        
        for folder in tqdm(matching_folders, desc="Processing folders"):
            logger.info(f"Processing folder: {folder.relative_to(root_folder)}")
            
            image_files = [f for f in folder.iterdir() 
                         if f.suffix.lower() in image_extensions]
            
            for file_path in tqdm(image_files, desc="Processing images", leave=False):
                relative_path = file_path.relative_to(root_folder)
                logger.info(f"Processing image: {relative_path}")
                
                try:
                    description = get_image_description(file_path, model, processor)
                    result = ProcessingResult(
                        description=description,
                        folder=str(folder.relative_to(root_folder)),
                        file_name=file_path.name
                    )
                except Florence2Error as e:
                    logger.error(f"Failed to process {relative_path}: {e}")
                    result = ProcessingResult(
                        description="",
                        folder=str(folder.relative_to(root_folder)),
                        file_name=file_path.name,
                        error=str(e)
                    )
                
                # Save result to both dictionary and CSV
                results[str(relative_path)] = result
                csv_writer.write_row(str(relative_path), result)
                
                # Periodic memory cleanup
                if len(results) % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
        
        return results
    finally:
        # Cleanup
        csv_writer.close()
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

def save_results_to_file(results, folder_path):
    """Save the results to a text file with detailed formatting."""
    output_file = Path(folder_path) / "detailed_image_descriptions_sans_texte.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Detailed Image Descriptions Report (Dossiers 'sans_texte')\n")
        f.write("=" * 80 + "\n\n")
        
        # Group results by folder
        folder_groups = {}
        for image_path, data in results.items():
            folder = data.folder
            if folder not in folder_groups:
                folder_groups[folder] = []
            folder_groups[folder].append((image_path, data))
        
        # Write results grouped by folder
        for folder, images in folder_groups.items():
            f.write(f"FOLDER: {folder}\n")
            f.write("-" * 80 + "\n\n")
            
            for image_path, data in images:
                f.write(f"File: {data.file_name}\n")
                f.write(f"Path: {image_path}\n")
                f.write("Description:\n{}\n".format(data.description))
                f.write("-" * 80 + "\n\n")
            
            f.write("\n")
    
    return output_file

def main():
    try:
        folder_path = Path("output_images")
        logger.info(f"Starting image processing in: {folder_path}")
        
        if not folder_path.exists():
            raise Florence2Error(f"Folder not found: {folder_path}")
        
        results = process_image_folder(folder_path)
        
        if not results:
            logger.warning("No images found in folders ending with 'sans_texte'!")
            return
        
        # Save text output (CSV is already saved incrementally)
        txt_file = save_results_to_file(results, folder_path)
        csv_file = folder_path / "image_descriptions_sans_texte.csv"
        
        logger.info(f"Results saved to text file: {txt_file}")
        logger.info(f"Results saved to CSV file: {csv_file}")
        
        # Print processing statistics
        successful = sum(1 for r in results.values() if not r.error)
        failed = sum(1 for r in results.values() if r.error)
        logger.info(f"Processing complete. Successful: {successful}, Failed: {failed}")
        
    except Florence2Error as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()