import os
import sys
# TODO this should be actually replaced by installing the module
sys.path.insert(0, os.path.abspath('.'))

from src.main_driver import MainDriver
from data.natural_instructions_dataset.dataset import DataHandlerNaturalInstructions

# TODO move here documentation, arguments and checks exclusive of this experiment
def main():
    driver = MainDriver(data_handler_class=DataHandlerNaturalInstructions)
    driver.run()


if __name__ == "__main__":
    main()
