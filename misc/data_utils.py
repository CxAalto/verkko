# Functions and classes for simple data handling
from math import sqrt, floor, ceil
import operator
import sys
import numpy as np

def read_columns(filename, header_rows, types, columns = None, sep=None):
    """Read columns of a file into lists

    Read the columns in file into lists, ignoring headers, and
    converting the values in column i to type types[i]. By default the
    len(types) first columns are read, if other columns are desired
    they must be specified explisitely with the argument
    'columns'. Note that the first column is 0. Filename can be either
    a string or a file object.

    Parameters
    ----------
    filename : string or a file object
        The input file to read. Note that if a file object is given,
        the caller is responsible for closing the file.
    header_rows : int
        The number of header rows in the files to skip before the
        actual data begins.
    types : tuple of type objects
        The types of the columns to be read.
    columns : sequence (with same length as types)
        The columns that are to be read, the first column being 0. If
        None, the first len(types) columns will be read.
    sep : string
        The column separator, defaults to any whitespace.

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
    >>> inputFile = open('input.txt')
    >>> col1, col4 = read_columns(inputFile, 2, (int, float), (1,4))
    >>> inputFile.close()

    >>> # Read only the second column. Note the use of tuples!
    >>> col1, = read_columns('input.txt', 2, (int,), (1,))

    
    """
    # If columns are not specified use the first columns.
    if columns == None:
        columns = range(len(types))

    # Column number must match type number.
    if len(columns) != len(types):
        raise Exception, "Number of types does not match the number of columns."

    # Open file
    if isinstance(filename, str):
        try:
            f = open(filename, 'rU')
        except IOError:
            raise
    elif isinstance(filename, file):
        f = filename

    # Read through headers        
    for i in range(header_rows):
        f.next()
        
    data = [[] for i in range(len(columns))]
    converted_fields = [0 for i in range(len(columns))]

    for line_number, line in enumerate(f):
        fields = line.split(sep)
        i, col = 0, 0
        try:
            for i, col in enumerate(columns):
                converted_fields[i] = types[i](fields[col])
        except ValueError:
            raise ValueError("Line %d, column %d: "
                             "Unable to convert '%s' to %s." % 
                             (line_number+header_rows+1,col,fields[col],
                              str(types[i])))
        except IndexError:
            raise IndexError("Line %d: Column %d not found." 
                             % (line_number+header_rows+1, col))

        # Line read successfully, add to data.
        for i in range(len(columns)):
            data[i].append(converted_fields[i])

    if isinstance(filename, str):
        f.close()
        
    return data



def gen_columns(filename, header_rows, types, columns = None, sep=None):
    """Generate column data from a file

    This function works exactly as read_columns, but instead of
    reading the whole file and returning the columns as lists, this
    functions is a generator and at each iteration yields a tuple with
    the values of each column.

    This function is especially handy when used with the Bins class
    for binning data straight from a file, because Bins can take a
    generator as input data.

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
    >>> inputFile = open('input.txt')
    >>> for (col1, col4) in gen_columns(inputFile, 2, (int, float), (1,4)):
            print '%d: %d' % (col1, col4) 
    >>> inputFile.close()
    
    """
    # If columns are not specified use the first columns.
    if columns == None:
        columns = range(len(types))

    # Column number must match type number.
    if len(columns) != len(types):
        raise Exception, "Number of types does not match the number of columns."

    # Open file
    if isinstance(filename, str):
        try:
            f = open(filename, 'rU')
        except IOError:
            raise
    elif isinstance(filename, file):
        f = filename

    # Read through headers
    for i in range(header_rows):
        f.next()
        
    for line_number, line in enumerate(f):
        converted_fields = []
        fields = line.split(sep)
        i, col = 0, 0
        try:
            for i, col in enumerate(columns):
                converted_fields.append( types[i](fields[col]) )
        except ValueError:
            raise ValueError("Line %d, column %d: "
                             "Unable to convert '%s' to %s." % 
                             (line_number+header_rows+1,col,fields[col],
                              str(types[i])))
        except IndexError:
            raise IndexError("Line %d: Column %d not found." 
                             % (line_number+header_rows+1, col))

        # Line read successfully, give output.
        yield tuple(converted_fields)

    # Close file if filename is a string.
    if isinstance(filename, str):
        f.close()
        

def process_columns(filename, header_rows, fun):
    """Read columns of a file by filtering lines with fun.

    Read the columns in file into lists, ignoring headers, and process
    the values in each row with fun. The return values of fun are
    collected into lists. Filename can be either a string or a file
    object.

    Parameters
    ----------
    filename : string or a file object
        The input file to read. Note that if a file object is given,
        the caller is responsible for closing the file.
    header_rows : int
        The number of header rows in the files to skip before the
        actual data begins.
    fun : function
        Function for processing a single line. fun takes in a list of
        strings (values in each column) and outputs the processed
        data. The n:th list in the output of process_columns will
        consist of the n:th elements of the outputs of fun.

    Return
    ------
    data : tuple of lists
        The n:th list contains all n:th return values of fun.

    Examples
    --------
    >>> # Sum columns 0 and 2, multiply columns 1 and 3.
    >>> # Exclude the two first rows (headers).
    >>> def myFun(cols):
    ...     fields = map(int, cols)
    ...     return fields[0]+fields[2], fields[1]*fields[3]
    ...
    >>> sum02, mult13 = process_columns('input.txt', 2, myFun)

    """

    # Open file
    if isinstance(filename, str):
        try:
            f = open(filename, 'rU')
        except IOError:
            raise
    elif isinstance(filename, file):
        f = filename

    # Read through headers        
    for i in range(header_rows):
        f.next()

    # Read first line to find out the return type of fun.
    fun_output = fun(f.next().split())
    wrapper = lambda x: (x,)
    output_len = 1
    if isinstance(fun_output, (tuple, list)):
        wrapper = lambda x: x
        output_len = len(fun_output)

    # Add the first read line.
    data = [[x] for x in wrapper(fun_output)]
    
    for line_number, line in enumerate(f):
        data_out = wrapper(fun(line.split()))
        for i in range(output_len):
            data[i].append(data_out[i])

    if isinstance(filename, str):
        f.close()
        
    return data


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
