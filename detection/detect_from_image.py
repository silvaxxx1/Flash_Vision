import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import requests
import glob
from datetime import datetime
from io import BytesIO
import argparse

from object_detection.utils import visualization_utils as vis_utils
from object_detection.utils import label_map_util

def load_model(model_path):
    """
    Load the TensorFlow model from the given path.
    
    Args:
        model_path (str): Path to the saved model directory.

    Returns:
        function: Loaded model's inference function.
    """
    model = tf.saved_model.load(model_path)
    return model.signatures['serving_default']

def path_to_np(path):
    """
    Convert an image file path or URL to a NumPy array.
    
    Args:
        path (str): Image file path or URL.

    Returns:
        np.ndarray: Image as a NumPy array.
    """
    if path.startswith('http://') or path.startswith('https://'):
        response = requests.get(path)
        image = Image.open(BytesIO(response.content))
    else:
        img_data = tf.io.gfile.GFile(path, 'rb').read()
        image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def run_inference_on_single_image(model, image):
    """
    Run inference on a single image.
    
    Args:
        model (function): Loaded model's inference function.
        image (np.ndarray): Image as a NumPy array.

    Returns:
        dict: Dictionary containing detection results.
    """
    image_tensor = tf.convert_to_tensor(image)
    image_tensor = image_tensor[tf.newaxis, ...]

    output_dict = model(image_tensor)
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy() for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    return output_dict

def run_inference(model, category_index, image_paths, output_path):
    """
    Run inference on a list of images and save the output with visualized detections.
    
    Args:
        model (function): Loaded model's inference function.
        category_index (dict): Category index for label mapping.
        image_paths (list): List of image file paths or URLs.
        output_path (str): Path to save output images.
    """
    for image_path in image_paths:
        image_np = path_to_np(image_path)
        output = run_inference_on_single_image(model, image_np)

        vis_utils.visualize_boxes_and_labels_on_image_array(
            image_np,
            output['detection_boxes'],
            output['detection_classes'],
            output['detection_scores'],
            category_index,
            instance_masks=output.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            min_score_thresh=0.75
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_path, f"detection_output_{timestamp}_{os.path.basename(image_path)}.png")
        plt.imshow(image_np)
        plt.savefig(output_file)
        plt.close()

def main(args):
    """
    Main function to load the model, read the images, and run inference.
    
    Args:
        args (argparse.Namespace): Command line arguments.
    """
    detection_model = load_model(args.model)
    category_index = label_map_util.create_category_index_from_labelmap(args.labelmap, use_display_name=True)
    
    # Check if the image_path is a directory or a list of image paths
    if os.path.isdir(args.image_path):
        image_formats = ['*.jpg', '*.jpeg', '*.png']
        images = []
        for format in image_formats:
            images.extend(glob.glob(os.path.join(args.image_path, format)))
    else:
        images = args.image_path.split(',')
    
    # Run inference on the images
    run_inference(detection_model, category_index, images, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detection from images')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model Path')
    parser.add_argument('-l', '--labelmap', type=str, required=True, help='Path to Label Map')
    parser.add_argument('-i', '--image_path', type=str, required=True, help='Path to Input Images')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Path to Output Directory')

    args = parser.parse_args()
    main(args)

# make sure to modify this according to the path of TF2 Object Detction API 
# python detect_from_image.py -m C:\Users\USER\SILVA\TF2_CV_Repo\models\MyModelHub\faster_rcnn_resnet50_coco_2018_01_28\saved_model -l C:\Users\USER\SILVA\TF2_OD_API\models\research\object_detection\data\mscoco_label_map.pbtxt -i New_folder -o C:\Users\USER\SILVA\TF2_CV_Repo\detection\outputs