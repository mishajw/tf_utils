from datetime import datetime
from typing import Optional
import os
import tensorflow as tf
import logging


log = logging.getLogger(__name__)


def add_arguments(parser):
    parser.add_argument(
        "--model_checkpoint_directory", type=str, default="model_checkpoints", env_var="MODEL_CHECKPOINT_DIRECTORY")
    parser.add_argument("--initial_model_checkpoint", type=str, default=None, env_var="INITIAL_MODEL_CHECKPOINT")


class ModelSaver:
    def __init__(self, model_checkpoint_directory: str, initial_model_checkpoint: Optional[str]):
        self.model_checkpoint_directory = model_checkpoint_directory
        self.base_model_checkpoint = initial_model_checkpoint

        self.saver = tf.train.Saver()
        self.start_time_formatted = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_checkpoint_path = os.path.join(
            self.model_checkpoint_directory, "model_%s.ckpt" % self.start_time_formatted)

        self.model_cost = None

    @classmethod
    def from_arguments(cls, args) -> "ModelSaver":
        return ModelSaver(args.model_checkpoint_directory, args.initial_model_checkpoint)

    def load(self, session: tf.Session):
        """
        Load the model from a file
        :param session: the current model
        """
        if self.base_model_checkpoint is None:
            log.debug("No base model checkpoint to read from")
            return

        try:
            self.saver.restore(session, self.base_model_checkpoint)
            log.debug("Restored model from %s" % self.base_model_checkpoint)
        except IOError:
            log.exception("Couldn't restore model because of IO error")

    def save(self, session: tf.Session, model_cost: float, ignore_cost: bool=False):
        """
        Save the current model if it is the best recorded one
        :param session: the current session
        :param model_cost: the current cost of the model
        :param ignore_cost: ignore the current cost and save the model regardless
        """
        if not ignore_cost and self.model_cost is not None and self.model_cost < model_cost:
            log.debug("Not writing out model because it is not better than best written model")
            return

        self.saver.save(session, self.model_checkpoint_path)
        log.debug("Saved model to %s" % self.model_checkpoint_path)
