## STEPS TO COMPLETE ##
# 1. import logging
# 2. set up config file for logging called `results.log`
# 3. add try except with logging for success or error
#    in relation to checking the types of a and b
# 4. check to see that log is created and populated correctly
#    should have error for first function and success for
#    the second
# 5. use pylint and autopep8 to make changes
#    and receive 10/10 pep8 score
""" Add values and return them"""

import logging

logging.basicConfig(
    filename="./exercise_results.log",
    level=logging.INFO,
    filemode="w",
    format="%(asctime)s %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def sum_vals(first_value, second_value):
    """
    Args:
        first_value: (int)
        second_value: (int)
    Return:
        first_value + second_value (int)
    """
    try:
        logging.info(f"{first_value}, {second_value}")
        assert isinstance(first_value, int)
        assert isinstance(second_value, int)
        logging.info(f"SUCCESS:Variables are correct type")
        return first_value + second_value
    except AssertionError:
        logging.error("ERROR: Variables are of not right type")


if __name__ == "__main__":
    sum_vals("no", "way")
    sum_vals(4, 5)
