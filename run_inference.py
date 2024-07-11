##### RUN INFERENCE : 

import argparse
import logging
import os
from Layoutlmv3_inference.ocr import prepare_batch_for_inference
from Layoutlmv3_inference.inference_handler import handle

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
        parser.add_argument("--images_path", type=str, required=True, help="Path to the images directory")
        parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output JSON and annotated images")
        args = parser.parse_args()

        images_path = args.images_path
        image_files = os.listdir(images_path)
        images_path = [os.path.join(images_path, image_file) for image_file in image_files if image_file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        inference_batch = prepare_batch_for_inference(images_path)
        handle(inference_batch, args.model_path, args.output_dir)

    except Exception as err:
        os.makedirs('log', exist_ok=True)
        logging.basicConfig(filename='log/error_output.log', level=logging.ERROR,
                            format='%(asctime)s %(levelname)s %(name)s %(message)s')
        logger.error(err)
