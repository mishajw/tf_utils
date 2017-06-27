import tensorflow as tf
from datetime import datetime


def add_arguments(parser):
    parser.add_argument("--training_steps", type=int, default=-1)
    parser.add_argument("--testing_step", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)


def run(
        args,
        name,
        get_batch_fn,
        testing_data,
        model_input,
        model_output,
        test_callback,
        train_evaluations=None,
        test_evaluations=None):

    """
    Train a model with a default training set up
    :param args: the command line arguments for specifying how to run
    :param name: the name of the project
    :param get_batch_fn: a function that takes a batch size, and returns a list training data of that length, in the
    form of a tuple of input and output data
    :param testing_data: the testing data, in the form of a tuple of input and output data
    :param model_input: the placeholder to feed the input of the model
    :param model_output: the placeholder to feed the output of the model
    :param train_evaluations: what to evaluate when training (will also evaluate summaries)
    :param test_evaluations: what to evaluate when testing (will also evaluate summaries)
    :param test_callback: function called with the results of the evaluations passed in `test_evaluations`
    """
    if train_evaluations is None:
        train_evaluations = []

    if test_evaluations is None:
        test_evaluations = []

    run_with_update_loop(
        name,
        __get_default_update_loop(
            args,
            __get_default_update_step(
                args,
                get_batch_fn,
                testing_data,
                get_default_train_step(
                    model_input,
                    model_output,
                    train_evaluations),
                get_default_test_step(
                    model_input,
                    model_output,
                    test_evaluations,
                    test_callback))))


def run_with_update_loop(
        name,
        update_loop_fn):
    """
    Train a model with a callback for all updates
    :param name: the name of the project
    :param update_loop_fn: a function that updates the model continuously
    """

    # Set up session
    session = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Current update iteration
    step = 0

    # Set up writers
    time_formatted = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_writer = tf.summary.FileWriter("/tmp/%s/%s/train" % (name, time_formatted), session.graph)
    test_writer = tf.summary.FileWriter("/tmp/%s/%s/test" % (name, time_formatted), session.graph)

    # Add summaries to items to evaluate
    all_summaries = tf.summary.merge_all()

    update_loop_fn(session, step, train_writer, test_writer, all_summaries)


def run_with_update_step(
        args,
        name,
        update_step_fn):

    """
    Train a model with an update step called at each training iteration
    :param args: 
    :param name: 
    :param update_step_fn: 
    """
    run_with_update_loop(
        name,
        __get_default_update_loop(
            args,
            update_step_fn))


def run_with_test_train_steps(
        args,
        name,
        get_batch_fn,
        testing_data,
        train_step_fn,
        test_step_fn):
    """
    Train a model and specify training and testing steps
    :param args: the command line arguments specifying how to run
    :param name: the name of the project
    :param get_batch_fn: function to get a batch of data
    :param testing_data: all available testing data
    :param train_step_fn: the training step
    :param test_step_fn: the testing step
    """

    update_loop_fn = __get_default_update_loop(
        args, __get_default_update_step(args, get_batch_fn, testing_data, train_step_fn, test_step_fn))

    run_with_update_loop(
        name,
        update_loop_fn)


def get_default_train_step(
        model_input,
        model_output,
        evaluations):

    def train_step(session, step, training_input, training_output, summary_writer, all_summaries):
        # Run training
        train_results = session.run(
            evaluations + [all_summaries],
            __get_feed_dict(model_input, model_output, training_input, training_output))

        # Add training summaries to writer if any exist
        if all_summaries is not None:
            summary_writer.add_summary(train_results[-1], step)

    return train_step


def get_default_test_step(
        model_input,
        model_output,
        evaluations,
        test_callback):

    def test_step(session, step, testing_input, testing_output, summary_writer, all_summaries):
        # Run model with test data
        test_results = session.run(
            evaluations + [all_summaries],
            __get_feed_dict(model_input, model_output, testing_input, testing_output))

        # Add testing summaries to writer if any exist
        if all_summaries is not None:
            summary_writer.add_summary(test_results[-1], step)

        # Call the test callback if it exists
        if test_callback is not None:
            test_callback(*test_results)

    return test_step


def __get_default_update_loop(
        args,
        update_step_fn):

    def update_loop(session, step, train_writer, test_writer, all_summaries):
        while True:
            update_step_fn(session, step, train_writer, test_writer, all_summaries)

            # Check if we've run out of steps (never run out of steps if limit is negative)
            if 0 <= args.training_steps < step:
                break
            else:
                step += 1

    return update_loop


def __get_default_update_step(
        args,
        get_batch_fn,
        testing_data,
        train_step_fn,
        test_step_fn):

    def update_step(session, step, train_writer, test_writer, all_summaries):
        training_input, training_output = get_batch_fn(args.batch_size)
        train_step_fn(session, step, training_input, training_output, train_writer, all_summaries)

        # If we're on a testing step...
        if step % args.testing_step == 0:
            testing_input, testing_output = testing_data
            test_step_fn(session, step, testing_input, testing_output, test_writer, all_summaries)

    return update_step


def __get_feed_dict(model_input, model_output, batch_input, batch_output):
    return dict(
        ({model_input: batch_input} if model_input is not None else {}),
        **({model_output: batch_output} if model_output is not None else {}))
