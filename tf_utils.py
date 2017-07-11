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


def remove_nans(t):
    return tf.where(tf.is_nan(t), tf.zeros_like(t), t)


def try_create_scoped_variable(*args, **kwargs):
    """
    Create a variable. If it has not been created before, it will be created. If it has, the original will be returned.
    :param args: the arguments to pass to `tf.get_variable`
    :param kwargs: the keyword arguments to pass to `tf.get_variable`
    :return: the variable
    """
    try:
        return tf.get_variable(*args, **kwargs)
    except ValueError:
        tf.get_variable_scope().reuse_variables()
        return tf.get_variable(*args, **kwargs)


def int_array_from_str(s: str):
    return [int(i_str.strip()) for i_str in s.split(",")]


def add_all_trainable_summaries():
    for variable in tf.trainable_variables():
        with tf.variable_scope(variable.name.replace(":", "_")):
            tensor_summary(variable)
