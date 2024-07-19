# Automated Pipeline for Inference and Fine-Tuning Using TensorFlow 2 Object Detection API

## Overview

This project provides an automated pipeline for both inference and fine-tuning using TensorFlow 2 (TF2) Object Detection API. The goal is to streamline the process of object detection by integrating all necessary steps into a cohesive and automated workflow.

### Inference Pipeline

1. **Download the Model**: Easily download pre-trained models to get started with inference quickly.
2. **Run Detection**: Perform object detection on various input types, including:
   - **Images**: Analyze static images for object detection.
   - **Videos**: Process video files for continuous object detection.
   - **Webcam**: Real-time object detection through webcam feed.

### Fine-Tuning Pipeline

1. **Download the Model**: Begin with a pre-trained model as a starting point for fine-tuning.
2. **Data Pipeline**:
   - **Load the Data**: Import image and annotation data.
   - **Annotate Data**: Label images with bounding boxes and class information.
   - **Convert Annotations**: Transform annotations from XML format to CSV.
   - **Create TFRecord**: For large datasets, generate TFRecord files for efficient training.
3. **Retrain Pipeline**:
   - **Load Data**: Convert data to NumPy arrays for training.
   - **Model Config**: Define and adjust the model configuration.
   - **Device Config**: Specify the computational device (CPU, GPU, TPU).
   - **Training Loops**: Implement training loops to fine-tune the model.

### Application Development

The fine-tuned model can be utilized to build applications using the `App` directory. This allows for the development of custom applications on top of the automated pipeline, making the process more efficient and streamlined.

### Making the Process Easier, Efficient, and Automatic

This project automates various aspects of object detection and model training, reducing manual intervention and streamlining the workflow. By integrating all necessary components into a single pipeline, users can achieve more efficient and effective results.

## Project Structure

- **App**: (TODO)
- **detection**: Contains scripts for running inference on images, videos, and webcam feeds.
- **model**: Includes model downloader and pre-trained models.
- **retrain**: Contains data pipeline scripts, training loop, and dataset management.
- **README**: This file.
- **requirements**: Dependencies required for the project.

## Getting Started

1. **Set Up the Environment**: 
   - Install the TensorFlow Object Detection API by following the instructions [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/).
   - Install the required Python packages:

     ```bash
     pip install -r requirements.txt
     ```

2. **Download a Model**: (Ensure you are in the correct directory or use the full path)
   ```bash
   python model_downloader.py MODEL_LINK ./MyModelHub
   # Example
   python model_downloader.py http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz ./MyModelHub
   ```

3. **Run Inference**:
   - On images:
     ```bash
     python detect_from_image.py -m path_to_model/saved_model -l path_to_labels/mscoco_label_map.pbtxt -i input_image -o path_to_output_file/outputs
     ```
   - On video:
     ```bash
     python detect_from_video.py -m path_to_model/saved_model -l path_to_labels/mscoco_label_map.pbtxt -i input_video.mp4 -o path_to_output_file/outputs
     ```
   - On webcam:
     ```bash
     python detect_from_webcam.py -m path_to_model/saved_model -l path_to_labels/mscoco_label_map.pbtxt -o path_to_output_file/outputs
     ```

4. **Fine-Tune the Model**:

   **Data Pipeline:**
   - Load the images:
     ```bash
     python download_images.py image_urls.txt path_to_dataset
     ```
   - Annotate images:
     ```bash
     labelimg.exe
     ```
   - Convert XML to CSV:
     ```bash
     python annotations_to_csv.py path_to_xml_folder path_to_csv_output
     ```
   - Convert CSV to TFRecord (for larger datasets):
     ```bash
     python preprocess_tfrecord.py path/to/your.tfrecord num_classes batch_size
     ```

   **Note**: Alternatively, you can automate the entire data pipeline:
   ```bash
   python data_pipeline.py url_file output_dir xml_folder output_csv tfrecord_output
   ```

5. **Fine-Tuning**:
   Run the training loop from the terminal with:
   ```bash
   python scripts/retrain.py --pipeline_config path/to/pipeline.config 
     --checkpoint_path path/to/model_checkpoints 
     --csv_file path/to/data_annotation.csv 
     --image_dir path/to/dataset 
     --class_labels LABEL1 LABEL2
     --num_classes NUM_CLASSES 
     --batch_size BATCH_SIZE 
     --num_batches NUM_BATCHES 
     --learning_rate LEARNING_RATE 
     --device cpu  # or gpu
   ```

6. **Build Applications**: Utilize the `App` directory to create custom applications based on the fine-tuned model.

