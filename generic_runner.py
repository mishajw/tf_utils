from . import data_holder
from . import model_saver
from datetime import datetime
from typing import Tuple, Optional, Callable, TypeVar, Generic, List, Any
import itertools
import tensorflow as tf
import tf_utils


BatchType = TypeVar("BatchType")


class GenericRunner(Generic[BatchType]):
    def __init__(
            self,
            name: str,
            training_steps: int,
            testing_step: int,
            batch_size: int,
            add_all_summaries: bool,
            run_tag: str):

        self.__name = name
        self.__training_steps = training_steps
        self.__testing_step = testing_step  # TODO: Rename
        self.__batch_size = batch_size
        self.__add_all_summaries = add_all_summaries
        self.__run_tag = run_tag

        # Methods for handling train/test data
        self.__get_batch_fn = None  # type: Optional[Callable[[int], BatchType]]
        self.__test_data = None  # type: Optional[BatchType]
        self.__get_feed_dict_fn = None  # type: Optional[Callable[[BatchType], dict]]

        # Used to save session model for this run
        self.__model_saver = None  # type: Optional[tf_utils.model_saver.ModelSaver]

        # Writers for summaries
        self.__train_writer = None  # type: Optional[tf.summary.FileWriter]
        self.__test_writer = None  # type: Optional[tf.summary.FileWriter]

        # Holds step functions
        self.__all_steps_fn = self.__default_all_steps
        self.__step_fn = self.__default_step
        self.__train_evaluations = None
        self.__test_evaluations = None
        self.__test_callback_fn = None
        self.__train_step_fn = None
        self.__test_step_fn = None

    def set_data(self, get_batch_fn: Callable[[int], BatchType], test_data: BatchType):
        self.__get_batch_fn = get_batch_fn
        self.__test_data = test_data

    def set_data_holder(self, _data_holder: data_holder.DataHolder):
        self.__get_batch_fn = _data_holder.get_batch
        self.__test_data = _data_holder.get_test_data()

    def set_get_feed_dict(self, get_feed_dict_fn: Callable[[BatchType], dict]):
        self.__get_feed_dict_fn = get_feed_dict_fn

    def set_model_input_output(self, model_input: tf.Tensor, model_output: tf.Tensor):
        self.__get_feed_dict_fn = lambda inputs, outputs: {model_input: inputs, model_output: outputs}

    def set_optimizer(self, optimizer: tf.train.Optimizer):
        self.set_train_evaluations([optimizer])

    def set_train_evaluations(self, train_evaluations: List[Any]):
        self.__train_evaluations = train_evaluations
        self.__train_step_fn = self.__default_train_step

    def set_test_evaluations(self, test_evaluations: List[tf.Tensor]):
        self.__test_evaluations = test_evaluations
        self.__test_step_fn = self.__default_test_step

    def set_test_callback(self, test_callback_fn: Callable[[Any], None]):
        self.__test_callback_fn = test_callback_fn

    def set_train_step(self, train_step_fn):
        self.__train_step_fn = train_step_fn

    def set_test_step(self, test_step_fn):
        self.__test_step_fn = test_step_fn

    def set_model_saver(self, _model_saver: model_saver.ModelSaver):
        self.__model_saver = _model_saver

    def run(self):
        # Set up session
        with tf.Session() as session:
            tf.global_variables_initializer().run()

            if self.__model_saver is not None:
                self.__model_saver.load(session)

            # Add summaries for all trainable variables
            if self.__add_all_summaries:
                tf_utils.add_all_trainable_summaries()

            # Set up writers
            self.__train_writer, self.__test_writer = self.get_writers(session, self.__name, self.__run_tag)

            self.__all_steps_fn(session)

    def __default_all_steps(self, session: tf.Session):
        assert self.__step_fn is not None

        for step in itertools.count():
            self.__step_fn(session, step)

            if self.__training_steps is not None and step > self.__training_steps:
                break

    def __default_step(self, session: tf.Session, step: int):
        assert self.__get_batch_fn is not None
        assert self.__train_step_fn is not None
        assert self.__test_data is not None
        assert self.__test_step_fn is not None

        summaries = tf.summary.merge_all()

        train_batch = self.__get_batch_fn(self.__batch_size)
        self.__train_step_fn(session, step, train_batch, summaries, self.__train_writer)

        if step % self.__testing_step == 0:
            test_cost = self.__test_step_fn(session, step, self.__test_data, summaries, self.__test_writer)

            if self.__model_saver is not None:
                self.__model_saver.save(session, test_cost)

    def __default_train_step(
            self,
            session: tf.Session,
            step: int,
            batch: BatchType,
            summaries: tf.Tensor,
            summary_writer: tf.summary.FileWriter):

        assert self.__train_evaluations is not None

        # Run training
        train_results = session.run(
            self.__train_evaluations + [summaries],
            self.__get_feed_dict_fn(batch))

        # Add training summaries to writer if any exist
        if summaries is not None:
            train_results, summaries_result = train_results
            summary_writer.add_summary(summaries_result, step)

    def __default_test_step(
            self,
            session: tf.Session,
            step: int,
            batch: BatchType,
            summaries: tf.Tensor,
            summary_writer: tf.summary.FileWriter):

        # Run testing
        cost_result, *test_results = session.run(
            self.__test_evaluations + [summaries],
            self.__get_feed_dict_fn(batch))

        # Add testing summaries to writer if any exist
        if summaries is not None:
            *test_results, summaries_result = test_results
            summary_writer.add_summary(summaries_result, step)

        if self.__test_callback_fn is not None:
            self.__test_callback_fn(test_results)

        return cost_result

    @staticmethod
    def get_writers(
            session: tf.Session,
            name: str,
            run_tag: Optional[str] = None) -> Tuple[tf.summary.FileWriter, tf.summary.FileWriter]:
        """
        Get the test and train writers for writing summaries
        :param session: the session being used
        :param name: the name of the program
        :param run_tag: the tag of the run to be placed in the writer path
        :return: a tuple where item 1 is the train writer, and item 2 is the test writer
        """

        if run_tag is not None:
            name = "%s/%s" % (name, run_tag)

        time_formatted = datetime.now().strftime("%Y%m%d-%H%M%S")
        train_writer = tf.summary.FileWriter("/tmp/%s/%s/train" % (name, time_formatted), session.graph)
        test_writer = tf.summary.FileWriter("/tmp/%s/%s/test" % (name, time_formatted), session.graph)

        return train_writer, test_writer

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--training_steps", type=int, default=None)
        parser.add_argument("--testing_step", type=int, default=1000)
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--add_all_summaries", type=bool, default=False)
        parser.add_argument("--run_tag", type=bool, default=None)

    @classmethod
    def from_args(cls, args, name: str):
        return GenericRunner(
            name, args.training_steps, args.testing_step, args.batch_size, args.add_all_summaries, args.run_tag)
