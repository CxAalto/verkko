# Functions and classes for simple data handling
from math import sqrt, floor, ceil
import operator
import sys
import numpy as np
from random import sample
from collections import deque

class __field_converter(object):
    """Class for converting fields to given types."""

    def __init__(self, types, cols=None):
        # If columns are not specified use the first columns.
        if cols == None:
            cols = range(len(types))

        # Column number must match type number.
        if len(cols) != len(types):
            raise Exception, "Number of types does not match the number of columns."

        self.types = types
        self.cols = cols

    def __call__(self, fields):
        ret_val = []
        try:
            for i, col in enumerate(self.cols):
                ret_val.append(self.types[i](fields[col]))
        except ValueError:
            raise ValueError("Unable to convert '%s' to %s in column %d"
                             % (fields[col], str(self.types[i]), col))
        except IndexError:
            raise IndexError("Column %d not found." % (col,))
        return ret_val


def read_columns(filename, skip_rows, typesORfun, columns=None, sep=None):
    """Read columns of a file into lists.

    Read the columns in file `filename` into lists, ignoring the first
    `skip_rows` lines. This function has two different calling
    signatures, both of which have identical output. The first version
    simply converts each column to a given type, while the second
    allows specifying a function that is used to extract the column
    information from each line.

    The first call signature is
      read_columns(filename, skip_rows, types, columns=None, sep=None)
    This converts the values in column i to type `types[i]`. By
    default the len(types) first columns are read, if other columns
    are desired they must be specified explisitely with the argument
    'columns'. Note that the first column is 0.

    Parameters (whit column types)
    ----------
    filename : str
        The input file to read.
    skip_rows : int
        The number of header rows in the files to skip before the
        actual data begins.
    types : sequence of type objects
        The types of the columns to be read.
    columns : sequence (with same length as types)
        The columns that are to be read, the first column being 0. If
        None, the first len(types) columns will be read.
    sep : string
        The column separator, defaults to any whitespace.

    
    The second call signature is
      read_columns(filename, skip_rows, fun)
    This version calls the function `fun` for the fields on each line
    to obtain information of columns

    Parameters (with conversion function)
    ----------
    filename : str
        The input file to read.
    skip_rows : int
        The number of header rows in the files to skip before the
        actual data begins.
    fun : function
        Function for processing a single line. fun takes in a list of
        strings (values in each column) and outputs the processed data
        as a sequence. The n:th list in the output of process_columns
        will consist of the n:th elements of the outputs of fun.

    Return
    ------
    data : tuple of lists
        The data in each column.

    Except
    ------
    ValueError : Unable to convert a field to a given type.
    IndexError : The requested column is not found in the file.

    Examples
    --------
    >>> # Get the first two columns of file input.txt that contains
    >>> # two header rows. Both columns are of type int.
    >>> col0, col1 = read_columns('input.txt', 2, (int, int))

    >>> # Read columns 1 and 4 of input.txt and convert them to type
    >>> # int and float, respectively. Skip the two header rows.
    >>> col1, col4 = read_columns('input.txt', 2, (int, float), (1,4))

    >>> # Read only the second column. Note the use of tuples!
    >>> col1, = read_columns('input.txt', 2, (int,), (1,))

    >>> # Sum columns 0 and 2, multiply columns 1 and 3.
    >>> # Exclude the two first rows (headers).
    >>> def myFun(cols):
    ...     fields = map(int, cols)
    ...     return fields[0]+fields[2], fields[1]*fields[3]
    ...
    >>> sum02, mult13 = read_columns('input.txt', 2, myFun)
    """
    # Initialize data generator.
    data_gen = gen_columns(filename, skip_rows, typesORfun, columns, sep)

    # Get first columns to find out the number of colums.
    first_cols = data_gen.next()

    # Read the remaining columns.
    data = [[x] for x in first_cols]
    for cols in data_gen:
        for i, c in enumerate(cols):
            data[i].append(c)

    return data


def gen_columns(filename, skip_rows, typesORfun, columns = None, sep=None):
    """Generate column data from a file.

    This function works exactly as read_columns, but instead of
    reading the whole file and returning the columns as lists, this
    function yields a type a tuple with the values of each column on
    each iteration.

    This function is especially handy when used with the binner.Bins
    class for binning data straight from a file, because Bins can take
    a generator as input data.

    Parameters
    ----------
    (See function read_columns.)

    Yield
    -----
    column_data : tuple
        The values in each column of the current row.

    Except
    ------
    (See function read_columns.)

    Examples
    --------
    >>> # Print the values in the first two columns.
    >>> for values in gen_columns('input.txt', 2, (int, int)):
            print values

    >>> # Print the values in second and fifth columns.
    >>> for (col1, col4) in gen_columns('input.txt', 2, (int, float), (1,4)):
            print '%d: %d' % (col1, col4) 
    """
    # Open file and read through headers. If `filename` does not
    # exist, IOError will be raised.
    f = open(filename, 'rU')
    for i in range(skip_rows):
        f.next()

    if hasattr(typesORfun, '__getitem__'):
        # `typesORfun` argument is a sequence of types. Create a
        # callable class for conversions.
        fun = __field_converter(typesORfun, columns)
    elif hasattr(typesORfun, '__call__'):
        # `typesORfun` is a function (or a callable class).
        fun = typesORfun
    else:
        raise ValueError("Unidentified input values.")

    for line_no, line in enumerate(f):
        try:
            fun_output = fun(line.split())
        except ValueError as er:
            # Append line number to exception message.
            raise ValueError("Line %d: %s" % (line_no+skip_rows+1, str(er)))
        except IndexError as er:
            # Append line number to exception message.
            raise IndexError("Line %d: %s" % (line_no+skip_rows+1, str(er)))

        if fun_output == None:
            continue

        # Convert to tuple.
        try:
            fun_output = tuple(fun_output)
        except TypeError as er:
            raise TypeError("'fun' must return a sequence, got %s" % str(type(fun_output)))

        # Line read successfully, yield data.
        yield fun_output

    # Close the input file.
    f.close()
        

def percentile(data, p):
    """Calculate the p-percentile of data

    The p:th percentile is the value X for which p percent of the
    elements of data are smaller than X.  For instance with p = 0.5 this
    function returns the median.

    Parameters
    ----------
    data : sequence
        The input data. This must be sorted!
    p : float
        The percentile to calculate. 

    Return
    -------
    x : int or float
        The percentile. Returns an element of the data if p*(N-1) is
        an integer, otherwise x will be between two data values. If
        data is empty, returns None
    """
    if not data:
        return None # data = []
    
    i = float(p)*(len(data)-1)
    if i == int(i):
        return data[int(i)]

    i_lower = int(floor(i))
    i_upper = int(ceil(i))
    fraq = i - i_lower
    return (1-fraq)*data[i_lower] + fraq*data[i_upper]


def cumulative_dist(data, prob=None, format='descending'):
    """Create cumulative distribution from given data

    Parameters
    ----------
    data : sequence
        The input data whose cumulative distribution will be
        created. If prob=None, data may include the same value
        multiple times.
    prob : sequence
        If None, the distribution is created from data alone;
        otherwise prob[i] is used as the probability of value
        data[i]. The values in prob will be normalized, so plain
        counts may be used also instead of probabilities.
    format : string
        If 'descending', the returned cumulative distribution is P(x
        >= X). If 'ascenting', it will be P(x <= X).

    Return
    ------
    (value, cum_prob) : (list, list)
        Data points and their cumulative probabilities. The cumulative
        probability of value[i] is cum_prob[i]. value is sorted in
        ascenting order.

    Exceptions
    ----------
    ValueError : If format is not 'descending' or 'ascenting'.
    ValueError : If prob != None but its length does not match the
                 length of data.
    """

    # Make sure format is either 'descending' or 'ascenting'.
    if format not in ('descending', 'ascenting'):
        raise ValueError("The 'format' parameter must be either "
                         "'descending' or 'ascenting', got '"+str(format)+"'.")

    # Check the length of prob.
    if prob is not None and len(prob) != len(data):
        raise ValueError("The length of 'prob' does not match the "
                         "length of 'data'.")
    
    # If prob is not given, go through the data and create it. If prob
    # is given, sort it with data.
    if prob is None:
        data_dict = {}
        for x in data:
            data_dict[x] = data_dict.get(x,0) + 1
        deco = sorted(data_dict.items())
    else:
        deco = sorted(zip(data, prob))

    data = map(operator.itemgetter(0), deco)
    prob = map(operator.itemgetter(1), deco)

    # Turn prob into an array and normalize it.
    prob = np.array(prob, float)
    prob /= prob.sum()

    # Create cumulative distribution.
    cum_prob = np.cumsum(prob)
    if format == 'descending':
        cum_prob = list(1 - cum_prob)
        cum_prob = [1] + cum_prob[:-1]

    return (list(data), list(cum_prob))

if __name__ == '__main__':
    """Run unit tests if called."""
    from tests.test_data_utils import *
    unittest.main()
