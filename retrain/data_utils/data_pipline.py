from data_download import load_images
from annotations_to_csv import convert_annotations_to_csv
from create_tfrecord import convert_csv_to_tfrecord  # Corrected function name
import argparse

def run_data_pipeline(url_file, output_dir, xml_folder, output_csv, tfrecord_output):
    # Download images
    load_images(url_file, output_dir)
    
    # Convert annotations to CSV
    convert_annotations_to_csv(xml_folder, output_csv)
    
    # Create TFRecord
    convert_csv_to_tfrecord(output_csv, output_dir, tfrecord_output)  # Corrected function name

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the data pipeline for object detection.')
    parser.add_argument('url_file', type=str, help='Path to the file containing image URLs.')
    parser.add_argument('output_dir', type=str, help='Directory to save downloaded images.')
    parser.add_argument('xml_folder', type=str, help='Folder containing XML files.')
    parser.add_argument('output_csv', type=str, help='Output CSV file path.')
    parser.add_argument('tfrecord_output', type=str, help='Output TFRecord file path.')

    args = parser.parse_args()
    run_data_pipeline(args.url_file, args.output_dir, args.xml_folder, args.output_csv, args.tfrecord_output)
