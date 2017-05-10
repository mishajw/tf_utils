def add_arguments(parser):
    parser.add_argument("--cap_data", type=int, default=None)
    parser.add_argument("--testing_percentage", type=float, default=25)


class DataHolder:
    """
    Organises data into training and testing sets and allows batch access
    """

    @classmethod
    def from_input_output_lists(cls, args, input_list, output_list):
        assert len(input_list) == len(output_list)

        return DataHolder.from_input_output_pairs(args, list(zip(input_list, output_list)))

    @classmethod
    def from_input_output_pairs(cls, args, pairs):
        return cls(args, lambda i: pairs[i], len(pairs))

    def __init__(self, args, get_data_fn, data_length):
        """
        Initialise the data holder with a list of all the data
        :param args: the command line arguments for how to handle data
        :param get_data_fn: function that takes an index and returns the data at that index
        :param data_length: the length of the data
        """

        self.__get_data_fn = get_data_fn

        # Cap the data if provided in arguments
        if args.cap_data is not None:
            data_length = min(data_length, args.cap_data)

        # Set the training and testing data to be the correct sizes
        self.__num_training_data = int(float(data_length) * (1 - args.testing_percentage / 100.0))
        self.__num_testing_data = data_length - self.__num_training_data

        self.__batch_index = 0

    def get_batch(self, size):
        """
        Get the next batch of training data
        :param size: the size of the batch
        :return: the training data
        """

        # If the batch size is bigger than the training data size, return all training data
        if size > self.__num_training_data:
            return self.__unzip(self.__get_training_data_range(0, self.__num_training_data))

        training_data_left = self.__num_training_data - self.__batch_index

        # If we haven't got enough space left in the training data for the whole batch, loop around
        if training_data_left < size:
            batch = \
                self.__get_training_data_range(self.__batch_index, self.__num_training_data) + \
                self.__get_training_data_range(0, size - training_data_left)
            self.__batch_index = size - training_data_left
            return self.__unzip(batch)

        # Otherwise, return the next batch
        batch = self.__get_training_data_range(self.__batch_index, self.__batch_index + size)
        self.__batch_index += size
        return self.__unzip(batch)

    def get_test_data(self):
        """
        Get all testing data
        :return: testing data
        """
        return self.__get_testing_data_range(0, self.__num_testing_data)

    def __get_training_data_range(self, start, end):
        return self.__get_data_range(self.__get_training_data, start, end)

    def __get_testing_data_range(self, start, end):
        return self.__get_data_range(self.__get_testing_data, start, end)

    def __get_training_data(self, index):
        assert index >= 0
        assert index < self.__num_training_data

        return self.__get_data(index)

    def __get_testing_data(self, index):
        assert index >= 0
        assert index < self.__num_testing_data

        return self.__get_data(self.__num_testing_data + index)

    def __get_data(self, index):
        # TODO: Randomize data deterministically
        return self.__get_data_fn(index)

    @staticmethod
    def __get_data_range(get_data_fn, start, end):
        return [get_data_fn(i) for i in range(start, end)]

    @staticmethod
    def __unzip(data):
        # return tuple([list(part) for part in zip(*data)])

        inputs = []
        outputs = []

        for i, o in data:
            inputs.append(i)
            outputs.append(o)

        return inputs, outputs
