import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
import pandas as pd
import numpy as np
import os
import cv2
import argparse
import random

# Add this function to set the device
def set_device(device):
    if device.lower() == 'cpu':
        tf.config.set_visible_devices([], 'GPU')
        print("Using CPU")
    elif device.lower() == 'gpu':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            print("Using GPU")
        else:
            print("No GPU detected")
    elif device.lower() == 'tpu':
        # TPU setup can be added here if needed
        print("Using TPU (not implemented in this example)")

def load_model(pipeline_config, checkpoint_path, num_classes, device):
    with tf.device(device):
        # Load the configuration file into a dictionary
        configs = config_util.get_configs_from_pipeline_file(pipeline_config)
        model_config = configs["model"]

        # Adjust the model configuration
        model_config.ssd.num_classes = num_classes
        model_config.ssd.freeze_batchnorm = True

        detection_model = model_builder.build(
            model_config,
            is_training=True
        )

        tmp_box_predictor_checkpoint = tf.compat.v2.train.Checkpoint(
            _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
            _box_prediction_head=detection_model._box_predictor._box_prediction_head,
        )

        tmp_model_checkpoint = tf.compat.v2.train.Checkpoint(
            _feature_extractor=detection_model._feature_extractor,
            _box_predictor=tmp_box_predictor_checkpoint
        )

        tmp_model_checkpoint = tf.compat.v2.train.Checkpoint(model=tmp_model_checkpoint)
        tmp_model_checkpoint.restore(checkpoint_path).expect_partial()

        tmp_img, tmp_shape = detection_model.preprocess(tf.zeros([1, 640, 640, 3]))
        tmp_pred_dic = detection_model.predict(tmp_img, tmp_shape)
        tmp_detections = detection_model.postprocess(tmp_pred_dic, tmp_shape)

    return detection_model

def load_data(csv_file, image_dir, img_size=(640, 640)):
    # Load data from CSV
    data = pd.read_csv(csv_file)

    # Extract image paths and bounding boxes
    image_paths = data['filename'].values
    xmin = data['xmin'].values
    xmax = data['xmax'].values
    ymin = data['ymin'].values
    ymax = data['ymax'].values
    classes = data['class'].values

    train_images_np = []
    gt_boxes = np.stack([ymin, xmin, ymax, xmax], axis=-1)

    for image_path in image_paths:
        full_image_path = os.path.join(image_dir, image_path)
        image_np = cv2.imread(full_image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_np = cv2.resize(image_np, img_size)
        image_np = image_np.astype(np.float32) / 255.0
        train_images_np.append(image_np)

    return train_images_np, gt_boxes, classes

def preprocess_data(train_images_np, gt_boxes, classes, class_labels):
    num_classes = len(class_labels)
    gt_classes_one_hot_tensors = []
    label_id_offset = 1

    for class_name in classes:
        try:
            class_index = class_labels.index(class_name)
        except ValueError:
            print(f"Class '{class_name}' in CSV is not found in class_labels. Skipping...")
            continue

        zero_indexed_groundtruth_classes = np.ones(shape=[1], dtype=np.int32) * class_index
        gt_classes_one_hot_tensors.append(tf.one_hot(zero_indexed_groundtruth_classes, num_classes))

    train_image_tensors = []
    gt_box_tensors = []

    for train_image_np, gt_box_np in zip(train_images_np, gt_boxes):
        train_image_tensor = tf.expand_dims(tf.convert_to_tensor(
            train_image_np, dtype=tf.float32), axis=0)
        train_image_tensors.append(train_image_tensor)
        gt_box_tensor = tf.convert_to_tensor(gt_box_np.reshape(-1, 4), dtype=tf.float32)
        gt_box_tensors.append(gt_box_tensor)

    gt_classes_one_hot_tensors = [tf.convert_to_tensor(gt_class, dtype=tf.float32) for gt_class in gt_classes_one_hot_tensors]

    return train_image_tensors, gt_box_tensors, gt_classes_one_hot_tensors

@tf.function
def train_step_fn(image_list, groundtruth_boxes_list, groundtruth_classes_list, model, optimizer, vars_to_fine_tune):
    shapes = tf.constant([640, 640, 3], dtype=tf.int32)
    model.provide_groundtruth(
        groundtruth_boxes_list=groundtruth_boxes_list,
        groundtruth_classes_list=groundtruth_classes_list)
    with tf.GradientTape() as tape:
        preprocessed_image_tensor = tf.concat(
            [model.preprocess(image_tensor)[0]
             for image_tensor in image_list], axis=0)
        prediction_dict = model.predict(preprocessed_image_tensor, shapes)
        losses_dict = model.loss(prediction_dict, shapes)
        total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
        gradients = tape.gradient(total_loss, vars_to_fine_tune)
        optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
    return total_loss

def main(args):
    print(args)  # Print the parsed arguments for debugging

    tf.keras.backend.clear_session()
    set_device(args.device)

    detection_model = load_model(args.pipeline_config, args.checkpoint_path, args.num_classes, args.device)
    train_images_np, gt_boxes, classes = load_data(args.csv_file, args.image_dir)
    train_image_tensors, gt_box_tensors, gt_classes_one_hot_tensors = preprocess_data(train_images_np, gt_boxes, classes, args.class_labels)

    print('Start fine-tuning!', flush=True)
    optim = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=0.9)
    trainable_variables = detection_model.trainable_variables
    to_fine_tune = [var for var in trainable_variables if any([var.name.startswith(prefix) for prefix in args.prefixes_to_train])]

    for idx in range(args.num_batches):
        all_keys = list(range(len(train_images_np)))
        random.shuffle(all_keys)
        example_keys = all_keys[:args.batch_size]

        gt_boxes_list = [gt_box_tensors[key] for key in example_keys]
        gt_classes_list = [gt_classes_one_hot_tensors[key] for key in example_keys]
        image_tensors = [train_image_tensors[key] for key in example_keys]

        with tf.device(args.device):
            total_loss = train_step_fn(image_tensors, gt_boxes_list, gt_classes_list, detection_model, optim, to_fine_tune)

        if idx % 10 == 0:
            print('batch ' + str(idx) + ' of ' + str(args.num_batches) + ', loss=' +  str(total_loss.numpy()), flush=True)

    print('Done fine-tuning!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune an object detection model.')
    
    parser.add_argument('--pipeline_config', type=str, required=True, help='Path to pipeline config file.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to checkpoint.')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to CSV file with image annotations.')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images.')
    parser.add_argument('--class_labels', type=str, nargs='+', required=True, help='List of class labels.')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--num_batches', type=int, default=100, help='Number of batches.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--prefixes_to_train', type=str, nargs='+', default=[
        'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
        'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead'
    ], help='Prefixes of variables to train.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu', 'tpu'], help='Device to use for training.')
    
    args = parser.parse_args()
    main(args)

# run the retrain.py file (adjust the path accordin to your peoject srtucture)
""" python retrain.py  --pipeline_config C:\Users\USER\SILVA\TF2_CV_Repo\retrain\pipeline.config   
 --checkpoint_path C:/Users/USER/SILVA/TF2_CV_Repo/models/MyModelHub/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 
--csv_file C:\Users\USER\SILVA\TF2_CV_Repo\retrain\dataset\kendrick_lamar\output.csv   --image_dir C:\Users\USER\SILVA\TF2_CV_Repo\retrain\dataset\kendrick_lamar   
 --class_labels kendrick   --num_classes 1   --batch_size 16   --num_batches 50   --learning_rate 0.001   --device cpu"""