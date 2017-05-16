import tensorflow as tf


def create_generation_comparison_images(input_image, output_image, guess_image):
    """
    Takes the images from a generative model and display them side-by-side
    :param input_image: the image inputted into the model
    :param output_image: the truth image to be outputted by the model
    :param guess_image: the actual image outputted by the model
    :return: the images side-by-side
    """

    # Put all three images side by side
    all_images = tf.concat([input_image, guess_image, output_image], axis=2)

    # Limit to 0-255
    all_images = tf.maximum(0.0, tf.minimum(255.0, all_images))

    return all_images


def tensor_summary(t):
    t_mean = tf.reduce_mean(t)
    t_stddev = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(t, t_mean))))

    tf.summary.scalar("mean", t_mean)
    tf.summary.scalar("stddev", t_stddev)
    tf.summary.scalar("max", tf.reduce_max(t))
    tf.summary.scalar("min", tf.reduce_min(t))

