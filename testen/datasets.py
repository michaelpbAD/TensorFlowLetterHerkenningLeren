import tensorflow as tf
import numpy as np


# for image in image_filles:
#     decode_image = tf.image.decode_png(
#         image,
#         channels=0,         #Use the number of channels in the PNG-encoded image.
#         dtype= tf.uint8,    #An optional tf.DType from: tf.uint8, tf.uint16. Defaults to tf.uint8.
#         name="decode_png"   #optional
#     )
#     #Returns: A Tensor of type dtype. 3-D with shape [height, width, channels].
#
#     resized_image = tf.image.resize_images(decode_image, [128, 128])
#
#     imga_dataset = tf.data.Dataset.from_tensors(resized_image)




#https://gist.github.com/eerwitt/518b0c9564e500b4b50f
#https://kratzert.github.io/2017/06/15/example-of-tensorflows-new-input-pipeline.html


NUM_CLASSES = 62

def input_parser(img_path, label):
    # convert the label to one-hot encoding
    one_hot = tf.one_hot(label, NUM_CLASSES)

    # read the img from file
    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_file, channels=3)
    resized_image = tf.image.resize_images(img_decoded, [128, 128])


    return resized_image, one_hot




# Toy data
train_imgs = tf.constant(['train/img1.png', 'train/img2.png',
                          'train/img3.png', 'train/img4.png',
                          'train/img5.png', 'train/img6.png'])
train_labels = tf.constant([0, 0, 0, 1, 1, 1])

val_imgs = tf.constant(['val/img1.png', 'val/img2.png',
                        'val/img3.png', 'val/img4.png'])
val_labels = tf.constant([0, 0, 1, 1])

# create TensorFlow Dataset objects
tr_data = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels))
val_data = tf.data.Dataset.from_tensor_slices((val_imgs, val_labels))

#e.g. for the training dataset
tr_data = tr_data.map(input_parser)
