# File: test_mylibrary.py
# Pytest filename starts with "test_...."
import pytest
import logging

##################################
"""
Function to test
"""


def import_data(pth):
    df = pd.read_csv(pth)
    return df


##################################
"""
Fixture - The test function test_import_data() will 
use the return of path() as an argument
"""


@pytest.fixture(
    scope="module", params=["./data/bank_data.csv", "./data/hospital_data.csv"]
)
def path():
    value = request.param
    return value


##################################
"""
Test method
"""


def test_import_data(path):
    try:
        df = import_data(path)

    except FileNotFoundError as err:
        logging.error("File not found")
        raise err

    # Check the df shape
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0

    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns"
        )
        raise err
    pytest.df = df

    return df


# Test function
# See the `df = pytest.df` statement accessing the Dataframe object from Namespace
def test_function_two():
    df = pytest.df

    """
  Some assertion statements per your requirement.
  """


##################################


"""
Same thing with cache method:
"""

# Test function
# It uses the built-in request fixture
def test_import_data(request):
    try:
        df = import_data("./data/bank_data.csv")
    except FileNotFoundError as err:
        logging.error("File not found")
        raise err

    #    Some assertion statements per your requirement.

    request.config.cache.set("cache_df", df)
    return df

    # Test function


def test_function_two(request):
    df = request.config.cache.get("cache_df", None)
    """
  Some assertion statements per your requirement.
  """
