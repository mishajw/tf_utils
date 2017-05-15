import tensorflow as tf
from datetime import datetime


def add_arguments(parser):
    parser.add_argument("--training_steps", type=int, default=-1)
    parser.add_argument("--testing_step", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)


def run_with_defaults(
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

    run_with_train_loop(
        name,
        __get_default_train_loop(
            args,
            __get_default_train_step(
                args,
                get_batch_fn,
                testing_data,
                model_input,
                model_output,
                train_evaluations,
                test_evaluations,
                test_callback)))


def run_with_train_step(args, name, train_step_fn):
    """
    Train a model with a callback for a single training step
    :param args: the command line arguments specifying how to run
    :param name: the name of the project
    :param train_step_fn: the callback run at each step of training
    """
    run_with_train_loop(
        name,
        __get_default_train_loop(args, train_step_fn))


def run_with_train_loop(
        name,
        train_loop_fn):
    """
    Train a model with a callback for all training
    :param name: the name of the project
    :param train_loop_fn: a function that trains the model continuously
    """

    # Set up session
    session = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Current training iteration
    step = 0

    # Set up writers
    time_formatted = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_writer = tf.summary.FileWriter("/tmp/%s/%s/train" % (name, time_formatted), session.graph)
    test_writer = tf.summary.FileWriter("/tmp/%s/%s/test" % (name, time_formatted), session.graph)

    # Add summaries to items to evaluate
    all_summaries = tf.summary.merge_all()

    train_loop_fn(session, step, train_writer, test_writer, all_summaries)


def __get_default_train_loop(
        args,
        train_step_fn):

    def train_loop(session, step, train_writer, test_writer, all_summaries):
        while True:
            train_step_fn(session, step, train_writer, test_writer, all_summaries)

            # Check if we've run out of steps (never run out of steps if limit is negative)
            if 0 <= args.training_steps < step:
                break
            else:
                step += 1

    return train_loop


def __get_default_train_step(
        args,
        get_batch_fn,
        testing_data,
        model_input,
        model_output,
        train_evaluations,
        test_evaluations,
        test_callback):

    def train_step(session, step, train_writer, test_writer, all_summaries):
        # Get current batch for training
        batch_input, batch_output = get_batch_fn(args.batch_size)

        # Run training
        train_results = session.run(train_evaluations + [all_summaries], {
            model_input: batch_input,
            model_output: batch_output
        })

        # Add training summaries to writer if any exist
        if all_summaries is not None:
            train_writer.add_summary(train_results[-1], step)

        # If we're on a testing step...
        if step % args.testing_step == 0:
            # Get test data
            test_input, test_output = testing_data

            # Run model with test data
            test_results = session.run(test_evaluations + [all_summaries], {
                model_input: test_input,
                model_output: test_output
            })

            # Add testing summaries to writer if any exist
            if all_summaries is not None:
                test_writer.add_summary(test_results[-1], step)

            # Call the test callback if it exists
            if test_callback is not None:
                test_callback(*test_results)

    return train_step
