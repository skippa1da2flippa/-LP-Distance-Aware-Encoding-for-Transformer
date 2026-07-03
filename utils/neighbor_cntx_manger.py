from colorama import Fore, Style
from torch import Tensor


class NeighborContextManager:

    def __init__(self, source: Tensor, target: Tensor, log: bool = True):
        """Initialize the NeighborContextManager instance and store its configuration."""
        self.source: Tensor = source
        self.target: Tensor = target
        self.log: bool = log

    def __enter__(self) -> None:
        """Enter the context manager and prepare managed resources."""
        return None

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """Exit the context manager and release managed resources."""
        if exc_type is not None:
            if self.log:
                print(
                    "Source node " + Fore.RED + f"{self.source}" + Style.RESET_ALL +
                    " does not have any path connecting with node " + Fore.GREEN +
                    f"{self.target}" + Style.RESET_ALL + ", it will be left to infinity"
                )

            return True
