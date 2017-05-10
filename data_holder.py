def add_arguments(parser):
    parser.add_argument("--cap_data", type=int, default=None)
    parser.add_argument("--testing_percentage", type=float, default=25)


class DataHolder:
    """
    Organises data into training and testing sets and allows batch access
    """

    @classmethod
    def from_input_output_lists(cls, args, input_list, output_list):
        return cls(args, list(zip(input_list, output_list)))

    @classmethod
    def from_input_output_pairs(cls, args, pairs):
        return cls(args, pairs)

    def __init__(self, args, all_data):
        """
        Initialise the data holder with a list of all the data
        :param args: the command line arguments for how to handle data
        :param all_data: a complete list of data available
        """

        data_length = len(all_data)

        # Cap the data if provided in arguments
        if args.cap_data is not None:
            data_length = min(data_length, args.cap_data)

        # Set the training and testing data to be the correct sizes
        amount_of_training_data = int(float(data_length) * (1 - args.testing_percentage / 100.0))
        self.__training_data = all_data[:amount_of_training_data]
        self.__testing_data = all_data[amount_of_training_data:data_length]

        self.__batch_index = 0

    def get_batch(self, size):
        """
        Get the next batch of training data
        :param size: the size of the batch
        :return: the training data
        """

        # If the batch size is bigger than the training data size, return all training data
        if size > len(self.__training_data):
            return self.unzip(self.__training_data)

        training_data_left = len(self.__training_data) - self.__batch_index

        # If we haven't got enough space left in the training data for the whole batch, loop around
        if training_data_left < size:
            batch = self.__training_data[self.__batch_index:] + self.__training_data[:size - training_data_left]
            self.__batch_index = size - training_data_left
            return self.unzip(batch)

        # Otherwise, return the next batch
        batch = self.__training_data[self.__batch_index:self.__batch_index + size]
        self.__batch_index += size
        return self.unzip(batch)

    def get_test_data(self):
        """
        Get all testing data
        :return: testing data
        """
        return self.__testing_data

    @staticmethod
    def unzip(data):
        # return tuple([list(part) for part in zip(*data)])

        inputs = []
        outputs = []

        for i, o in data:
            inputs.append(i)
            outputs.append(o)

        return inputs, outputs
