import tensorflow as tf
import pandas as pd
import os
import argparse

def create_tf_example(row, image_dir):
    """
    Create a tf.train.Example from a row of the CSV file.

    Args:
        row (pd.Series): A row from the annotations CSV.
        image_dir (str): Directory where images are stored.

    Returns:
        tf.train.Example: A tf.train.Example containing the image and its annotations.
    """
    # Read the image file
    img_path = os.path.join(image_dir, row['filename'])
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_img = fid.read()

    # Create a dictionary with the features
    feature = {
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_img])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[row['filename'].encode('utf-8')])),
        'image/object/class/label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[row['class'].encode('utf-8')])),
        'image/object/bbox/xmin': tf.train.Feature(int64_list=tf.train.Int64List(value=[row['xmin']])),
        'image/object/bbox/ymin': tf.train.Feature(int64_list=tf.train.Int64List(value=[row['ymin']])),
        'image/object/bbox/xmax': tf.train.Feature(int64_list=tf.train.Int64List(value=[row['xmax']])),
        'image/object/bbox/ymax': tf.train.Feature(int64_list=tf.train.Int64List(value=[row['ymax']])),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

def convert_csv_to_tfrecord(csv_input, image_dir, output_path):
    """
    Convert CSV annotations to a TFRecord file.

    Args:
        csv_input (str): Path to the CSV file.
        image_dir (str): Directory where images are stored.
        output_path (str): Path to the output TFRecord file.
    """
    df = pd.read_csv(csv_input)
    writer = tf.io.TFRecordWriter(output_path)

    for _, row in df.iterrows():
        tf_example = create_tf_example(row, image_dir)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print(f"TFRecord file saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert CSV annotations to TFRecord.')
    parser.add_argument('csv_input', type=str, help='Path to the CSV file.')
    parser.add_argument('image_dir', type=str, help='Directory where images are stored.')
    parser.add_argument('output_path', type=str, help='Output TFRecord file path.')

    args = parser.parse_args()
    convert_csv_to_tfrecord(args.csv_input, args.image_dir, args.output_path)

if __name__ == '__main__':
    main()
