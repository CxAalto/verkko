# Functions and classes for simple data handling
from math import sqrt, floor, ceil
import operator
import sys

def read_columns(filename, header_rows, types, columns = None, sep=None):
    """ Read columns of a file into lists

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
            print "read_columns: Error reading line " + str(line_number + header_rows + 1) + \
                  ", column " + str(col) + ":"
            print "              Unable to convert " + fields[col] + " to " + str(types[i])
            exit()
        except IndexError:
            print "read_columns: Error reading line " + str(line_number + header_rows + 1) + ":"
            print "              Column " + str(col) + " not found."
            exit()
            
        # Line read successfully, add to data.
        for i in range(len(columns)):
            data[i].append(converted_fields[i])

    if isinstance(filename, str):
        f.close()
        
    return data



def gen_columns(filename, header_rows, types, columns = None, sep=None):
    """Generates tuples with values corresponding to those in columns.

    Read the given columns in given file, ignoring the first
    header_rows lines, and converting the values in column i to type
    types[i]. A tuple containing the values is return at each
    iteration.  By default the len(types) first columns are read,
    other columns must be specified explisitely (the first column is
    0). Filename can be either a string or a file object.

    Examples:
    >>> for values in gen_columns('input.txt', 2, (int, int)):
            print values

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
            print ("read_columns: Error reading line %d, column %d:\n" +
                   "              Unable to convert %s to %s"
                   % (line_number + header_rows + 1, col,
                      fields[col], str(types[i])) )
            exit()
        except IndexError:
            print ("read_columns: Error reading line %d:\n" +
                   "              Column %d not found."
                   % (line_number + header_rows + 1, col))
            exit()
            
        # Line read successfully, give output.
        yield tuple(converted_fields)

    if isinstance(filename, str):
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

