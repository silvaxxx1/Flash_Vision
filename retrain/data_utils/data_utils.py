import tensorflow as tf
import argparse

# Feature description dictionary
feature_description = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/object/class/label': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.FixedLenFeature([], tf.int64),
    'image/object/bbox/xmax': tf.io.FixedLenFeature([], tf.int64),
    'image/object/bbox/ymin': tf.io.FixedLenFeature([], tf.int64),
    'image/object/bbox/ymax': tf.io.FixedLenFeature([], tf.int64),
}

def parse_tfrecord_fn(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)

def decode_and_preprocess(parsed_example, num_classes):
    image = tf.image.decode_jpeg(parsed_example['image/encoded'])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [640, 640])
    
    xmin = tf.cast(parsed_example['image/object/bbox/xmin'], tf.float32) / image.shape[1]
    xmax = tf.cast(parsed_example['image/object/bbox/xmax'], tf.float32) / image.shape[1]
    ymin = tf.cast(parsed_example['image/object/bbox/ymin'], tf.float32) / image.shape[0]
    ymax = tf.cast(parsed_example['image/object/bbox/ymax'], tf.float32) / image.shape[0]
    boxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    
    class_text = parsed_example['image/object/class/label']
    classes = tf.where(class_text == b'kendrick', 1, 0)
    classes = tf.one_hot(classes, depth=num_classes)
    
    return image, boxes, classes

def create_dataset(tfrecord_path, num_classes, batch_size):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
    dataset = parsed_dataset.map(lambda x: decode_and_preprocess(x, num_classes))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess TFRecord dataset for object detection.")
    parser.add_argument('tfrecord_path', type=str, help="Path to the TFRecord file.")
    parser.add_argument('num_classes', type=int, help="Number of classes for one-hot encoding.")
    parser.add_argument('batch_size', type=int, help="Batch size for the dataset.")
    
    args = parser.parse_args()
    
    dataset = create_dataset(args.tfrecord_path, args.num_classes, args.batch_size)
    
    # Optional: iterate through the dataset to print shapes of batches as a test
    for images, boxes, classes in dataset.take(1):
        print("Images shape:", images.shape)
        print("Boxes shape:", boxes.shape)
        print("Classes shape:", classes.shape)
