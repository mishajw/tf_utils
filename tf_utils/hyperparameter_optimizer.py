from typing import TypeVar, Generic, List, Any, Callable, Optional
import io
import math
import os
import skopt
import subprocess
import sys

ParameterType = TypeVar("ParameterType")


class OptimizableParameter(Generic[ParameterType]):
    def __init__(
            self,
            command_line_name: str,
            short_name: Optional[str],
            skopt_dimension: Any,
            default_value: ParameterType,
            value_to_string: Callable[[ParameterType], str] = str):
        self.command_line_name = command_line_name
        self.short_name = short_name
        self.skopt_dimension = skopt_dimension
        self.default_value = default_value
        self.value_to_string_fn = value_to_string

        if self.short_name is None:
            self.short_name = "".join(
                [word[0] for word in self.command_line_name.replace("-", "").split("_") if word != ""])

    def get_command_line_argument(self, value: ParameterType) -> List[str]:
        return [self.command_line_name, self.value_to_string_fn(value)]

    def get_run_tag(self, value: ParameterType) -> str:
        assert self.short_name is not None
        return f"{self.short_name}{self.value_to_string_fn(value)}"

    @classmethod
    def int_type(cls, command_line_name: str, short_name: Optional[str], min_value: int, max_value: int):
        return OptimizableParameter(command_line_name, short_name, (min_value, max_value), (max_value + min_value) / 2)

    @classmethod
    def float_type(cls, command_line_name: str, short_name: Optional[str], min_value: float, max_value: float):
        return OptimizableParameter(command_line_name, short_name, (min_value, max_value), (max_value + min_value) / 2)

    @classmethod
    def int_power_type(
            cls,
            command_line_name: str,
            short_name: Optional[str],
            min_value: int,
            max_value: int,
            base: int = 2):
        min_pow = int(math.log(min_value, base))
        max_pow = int(math.log(max_value, base))
        possible_values_str = [str(int(pow(base, p))) for p in range(min_pow, max_pow + 1)]

        return OptimizableParameter(
            command_line_name, short_name, possible_values_str, possible_values_str[int(len(possible_values_str) / 2)])

    @classmethod
    def bool_type(cls, command_line_name: str, short_name: Optional[str]):
        return OptimizableParameter(command_line_name, short_name, ["true", "false"], "true")

    @staticmethod
    def bind_with_parameters(optimizable_parameters: List["OptimizableParameter"], parameters: List[Any]):
        assert len(parameters) == len(optimizable_parameters)

        command_line_arguments = [optimizable.get_command_line_argument(p)
                                  for optimizable, p in zip(optimizable_parameters, parameters)]

        # Flatten the results
        return [part for cli in command_line_arguments for part in cli]

    @staticmethod
    def get_full_run_tag(optimizable_parameters: List["OptimizableParameter"], parameters: List[Any]) -> str:
        assert len(parameters) == len(optimizable_parameters)

        return "_".join([optimizable.get_run_tag(p) for optimizable, p in zip(optimizable_parameters, parameters)])


def create_function(base_command: List[str], optimizable_parameters: List[OptimizableParameter], log_directory: str):
    def run_with_parameters(parameters):
        print("Starting job")
        last_output = None

        run_tag = OptimizableParameter.get_full_run_tag(optimizable_parameters, parameters)
        with open(os.path.join(log_directory, run_tag), "w") as process_log_file:
            process = subprocess.Popen(
                base_command +
                OptimizableParameter.bind_with_parameters(optimizable_parameters, parameters) +
                ["--run_tag", run_tag],
                stdout=subprocess.PIPE)

            # Read stdout from the process
            for process_output in io.TextIOWrapper(process.stdout):
                # Write the logs to a file
                process_log_file.write(process_output)

                # Store the last output for cost parsing
                last_output = process_output

        print("Finished job")

        try:
            return float(last_output)
        except (ValueError, TypeError):
            print(f"Couldn't parse {last_output} as float for cost", file=sys.stderr)
            return None

    return run_with_parameters


def run(
        base_command: List[str],
        optimizable_parameters: List[OptimizableParameter],
        log_directory: str,
        num_concurrent: int):

    if not os.path.isdir(log_directory):
        os.mkdir(log_directory)

    run_with_parameters = create_function(base_command, optimizable_parameters, log_directory)

    skopt.gp_minimize(
        run_with_parameters,
        dimensions=[optimizable.skopt_dimension for optimizable in optimizable_parameters],
        acq_optimizer="lbfgs",
        n_jobs=num_concurrent,
        n_random_starts=0,
        x0=[p.default_value for p in optimizable_parameters],
        verbose=100)
