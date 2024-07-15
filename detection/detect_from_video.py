
import tensorflow as tf
import numpy as np
import cv2
import os
from datetime import datetime
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

def run_inference_on_video(model, category_index, input_video_path, output_path):
    """
    Run inference on a video file and save the output video.

    Args:
        model (function): Loaded model's inference function.
        category_index (dict): Category index for label mapping.
        input_video_path (str): Path to the input video file.
        output_path (str): Path to save output video.
    """
    cap = cv2.VideoCapture(input_video_path)  # Open video file

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_file = os.path.join(output_path, f"detection_output_{timestamp}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_file, fourcc, fps, frame_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

        output_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        out.write(output_bgr)

        # Optionally display the output frame
        cv2.imshow('Object Detection', output_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main(args):
    """
    Main function to load the model and run inference on a video file.

    Args:
        args (argparse.Namespace): Command line arguments.
    """
    detection_model = load_model(args.model)
    category_index = label_map_util.create_category_index_from_labelmap(args.labelmap, use_display_name=True)
    run_inference_on_video(detection_model, category_index, args.input_video, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detection from video file')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model Path')
    parser.add_argument('-l', '--labelmap', type=str, required=True, help='Path to Label Map')
    parser.add_argument('-i', '--input_video', type=str, required=True, help='Path to Input Video File')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Path to Output Directory')

    args = parser.parse_args()
    main(args)

# make sure to modify this according to the path of TF2 Object Detction API 
# python detect_from_video.py -m C:\Users\USER\SILVA\TF2_CV_Repo\models\MyModelHub\faster_rcnn_resnet50_coco_2018_01_28\saved_model -l C:\Users\USER\SILVA\TF2_OD_API\models\research\object_detection\data\mscoco_label_map.pbtxt -i C:\path\to\input_video.mp4 -o C:\Users\USER\SILVA\TF2_CV_Repo\detection\outputs
