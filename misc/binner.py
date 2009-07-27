#!/usr/bin/python2.6
#
# Lauri Kovanen, 2009 (lauri.kovanen@gmail.com)
# Department of Biomedical Engineering and Computational Science
# Helsinki University of Technology
"""Bin data into 1 or 2-dimensional bins.

   Bins: Bin data into 1-dimensional bins.
   Bins2D: Bin data into 1-dimensional bins.
   normalize(): Divide a sequence by its sum.
"""


from math import ceil, floor, sqrt
from operator import itemgetter
import numpy as np

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class ParameterError(Error):
    """Exception raised for errors in the parameter."""
    pass

class BinLimitError(Error):
    """Exception raised when bin limits are invalid."""
    pass

class DataTypeError(Error):
    """Exception raised when wrong kind of data is given."""
    pass
        
def normalize(x):
    """Normalize a sequence

    Returns the sequence where each element is divided by the sum of
    all elements.
    """
    return list(np.array(x,float)/sum(x))

class _binFinder(object):
    """
    Find the correct bin for a given value by searching through all
    bins with binary search. The correct bin is bin i with
    bin_limits[i] <= value < bin_limits[i+1].

    If value < bin_limits[0], the returned bin index is negative, and
    if value >= bin_limits[-1], the returned bin index is greater than
    len(bins)-2.

    Finding the correct bin takes log(N) time. If it is possible to
    find the correct bin in constant time (as is the case with for
    instance linear and logarithmic bins), you should inherit this
    class and reimplement the __init__ and __call__ methods.
    
    Parameters
    ----------
    bin_limits : sequence
        The limits of the bins used in binning. If there are N bins,
        len(bin_limits) is N+1.
    value : integer or float
        The value for which the correct bin should be located.

    Returns
    -------
    bin_index : integer
        The index of the bin for value.

    """

    def __init__(self, bin_limits):
        # Add auxilary bins to front and back. This way we can return
        # -1 if the value is below the true first bin limit and N if
        # the value is beyond the actual last bin.
        self.bin_limits = ( [bin_limits[0]-1] + list(bin_limits) +
                            [bin_limits[-1]+1] )

    def __call__(self, value):
        lo, hi = 0, len(self.bin_limits)-1
        while (lo < hi - 1):
            mid = (lo+hi)/2
            #print "bin_limits["+str(lo)+"] = ", str(self.bin_limits[lo]),
            #print "  bin_limits["+str(mid)+"] = ", str(self.bin_limits[mid]),
            #print "  bin_limits["+str(hi)+"] = ", str(self.bin_limits[hi])
            #raw_input("Press any key ...")
            if self.bin_limits[mid] <= value:
                lo = mid
            if self.bin_limits[mid] > value:
                hi = mid
        return lo - 1

class _linBinFinder(_binFinder):
    """
    Return the correct bin for a given value with linear bins.

    If value < bin_limits[0], the returned index is negative, and if
    value >= bin_limits[-1], the returned index is > len(bins)-2.

    Parameters
    ----------
    bin_limits : sequence
        The limits of the bins used in binning. If there are N bins,
        len(bin_limits) is N+1.
    value : integer or float
        The value for which the correct bin should be located.

    Returns
    -------
    bin_index : integer
        The index of the bin for value.
    """

    def __init__(self, bin_limits):
        self.minLimit = bin_limits[0]
        self.diff = float(bin_limits[-1] - self.minLimit)/(len(bin_limits) - 1)
        #print "lin self.diff:", repr(self.diff)
        
    def __call__(self, value):
        #print "0:", repr(value), repr(self.minLimit)
        #print "1:", repr(value - self.minLimit)
        #print "2:", repr((value - self.minLimit)/self.diff)
        #print "3:", repr(floor( (value - self.minLimit)/self.diff ))
        return int(floor((value - self.minLimit)/self.diff))
    
class _logBinFinder(_binFinder):
    """
    Return the correct bin for a given value with logarithmic bins.

    If value < bin_limits[0], the returned index is negative, and if
    value >= bin_limits[-1], the returned index is greater than
    len(bin_limits)-2.

    Parameters
    ----------
    bin_limits : sequence
        The limits of the bins used in binning. If there are N bins,
        len(bin_limits) is N+1.
    value : integer or float
        The value for which the correct bin should be located.

    Returns
    -------
    bin_index : integer
        The index of the bin for value.
    """

    def __init__(self, bin_limits):
        # Note that _logBinFinder is just _linBinFinder with logarithmic
        # bin_limits and values. However, that simple implementation
        # doesn't work because of floating point inaccuracies; with
        # floats there are cases when log(a)-log(b) != log(a/b).
        self.minLimit = bin_limits[0]
        self.N_bins = len(bin_limits)-1
        self.ratio = np.log2(float(bin_limits[-1])/bin_limits[0])
        #print "log2 self.ratio:", repr(self.ratio)
        
    def __call__(self, value):
        try:
            return int(floor(self.N_bins * np.log2(float(value)/self.minLimit)
                             / self.ratio ))
        except (OverflowError, ValueError):
            # value = 0 gives OverflowError, negative values give
            # ValueError.  Since these are always below the lowest bin
            # limit in a logarithmic bin, we return a negative index.
            return -1

class _linlogBinFinder(_binFinder):
    """
    Return the correct bin for a given value with linear-logarithmic
    bins: linear bins from minLimit to 10.5, and logarithmic bins from
    thereon.

    Parameters
    ----------
    bin_limits : sequence
        The limits of the bins used in binning. If there are N bins,
        len(bin_limits) is N+1.
    value : integer or float
        The value for which the correct bin should be located.

    Returns
    -------
    bin_index : integer
        The index of the bin for value.
    """

    def __init__(self, bin_limits):
        self.bin_limit = np.min(10.5, bin_limits[-1])
        self.N_lin_bins = int(self.bin_limit - bin_limits[0])
        # Create linear binner if linear bins exist.
        if self.N_lin_bins > 0:
            self.lin_bf = _linBinFinder(bin_limits[:self.N_lin_bins+1])
        else:
            self.lin_bf = None
        # Create logarithmic binner if logarithmic bins exist.
        if bin_limits[-1] > 10.5:
            self.log_bf = _logBinFinder(bin_limits[self.N_lin_bins:])
        else:
            self.log_bf = None
        
    def __call__(self, value):
        if value < self.bin_limit:
            return self.lin_bf(value)
        else:
            return self.N_lin_bins + self.log_bf(value)

class _BinLimits(tuple):
    """Class that represents the bin limits.

    Implemented as a tuple with additional methods to facilitate
    construction and getting the centers and widths of the bins.
    """

    @classmethod
    def __generateLinbins(cls, minValue, maxValue, N_bins):
        """Generate linear bins.

        The first bin is centered around self.minValue and the last bin is
        centered around self.maxValue.

        Parameters
        ----------
        N_bins : integer
            Number of bins to generate

        Return
        ------
        limit_seq : list
            The bin limits. len(limit_seq) = N_bins + 1
        """

        binwidth = (maxValue-minValue)/float(N_bins-1)
        bins = [minValue-binwidth/2.0]
        currvalue = bins[0]

        while currvalue < maxValue:
            currvalue=currvalue+binwidth
            bins.append(currvalue)

        return bins

    @classmethod
    def __generateLogbins(cls, minValue, maxValue, factor, uselinear=True):
        """Generate logarithmic bins.

        The upper bound of each bin is created by multiplying the
        lower bound with factor. The lower bound of the first bin
        chosen so that minValue is the mid point of the first
        bin. Bins are created until maxValue fits into the last
        bin.

        Parameters
        ----------
        factor : float
            The factor for increasing bin limits.
        uselinear : boolean (True)
            If True, linear bins will be used between
            min(1, minValue) and max(10, maxValue). Each
            linear bin will include one integer.

        Return
        ------
        limit_seq : list
            The bin limits. len(limit_seq) = N_bins + 1
        """

        if uselinear:
            bins = [i-0.5 for i in range(max(0, int(minValue)),
                                         min(12, int(maxValue)+2))]
            i = len(bins)
        else:
            bins=[ minValue*2.0/(1+factor) ]
            i=1

        while bins[i-1] <= maxValue:
            bins.append(bins[i-1]*factor)
            i+=1

        return bins

    @classmethod
    def __generateMaxLogbins(cls, minValue, maxValue, diff = 0.01):
        """Generate as many logarithmic bins as possible.

        Construct logarthmic bins from minValue to maxValue
        so that each bin contains at least one integer. The first bin
        will start at (minValue-diff) and the last bin will end at
        (maxValue+diff).
        
        Parameters
        ----------
        diff : float (0.01)
            Extra space between minimum and maximum value. A smaller
            value for diff gives a larger number of bins.

        Return
        ------
        limit_seq : list
            The bin limits. len(limit_seq) = N_bins + 1

        Notes
        -----
        The number of bins grows quickly when minValue is increased
        beyond 1. It is a good idea to check that the resulting bins
        are still suitable for your purpose.
        """
        # Initial values
        if minValue > maxValue:
            return []
        a = minValue - diff
        b = maxValue + diff
        max_bins = maxValue - minValue + 1
        if max_bins == 1:
            return [a, b]

        # Find the integer that diffs factor size.
        i, factor = 1, 0
        cmp_factor = sqrt((minValue+1)/a)
        while factor < cmp_factor and i < max_bins:
            factor = cmp_factor
            i += 1
            cmp_factor = ((minValue+i)/a)**(1.0/(i+1))

        # Calculate the correct number of bins and the exact factor so
        # that this number of bins is reached.
        N_bin = int(np.log(b/a)/np.log(factor))
        factor = (b/a)**(1.0/N_bin)

        # Create bins.
        bins = [a]
        for i in range(N_bin):
            bins.append(bins[i]*factor)
        return bins

    @classmethod
    def __check_parameters(cls, minValue, maxValue, binType, param):
        """Make sure the construction parameters are valid."""

        if minValue >= maxValue:
            raise BinLimitError("minValue must be larger than maxValue.")

        if binType in ('log', 'logarithmic', 'maxlog'):
            if minValue <= 0:
                # minValue must be strictly positive in logarithmic bins.
                raise BinLimitError("minValue must be strictly positive "
                                    "with '%s' bin type." % (binType,))

        if binType in ('linlog', 'linmaxlog'):
            # minValue must be positive in logarithmic bins.
            if minValue < 0:
                raise BinLimitError("minValue must be positive "
                                    "with '%s' bin type." % (binType,))

            # The lower limit in linlog types must be integer, and
            # also the upper limit if it is below 10.5.
            if minValue < 11 and not isinstance(minValue, int):
                raise BinLimitError("When minValue < 11, minValue must be integer "
                                    "with '%s' bin type." % (binType,))

            if maxValue < 11 and not isinstance(maxValue, int):
                raise BinLimitError("When maxValue < 11, maxValue must be integer "
                                    "with '%s' bin type." % (binType,))


        if binType in ('log', 'logarithmic', 'linlog'):
            if param <= 1:
                # factor must be larger than 1.
                raise ParameterError("factor (param) must be larger than"
                                     " 1 with '%s' bin type."%(binType,))

        if binType in ('maxlog', 'linmaxlog') and param != None:
            if (param <= 0 or param > 1):
                # diff must be in [0,1).
                raise ParameterError("diff (param) must be in open "
                                     "interval (0,1) with '%s' bin type."
                                     % binType)

        if binType in ('lin', 'linear'):
            if not isinstance(param, int) or param <= 0:
                raise ParameterError("Number of bins (param) must "
                                     "be a positive integer.")

        if binType == 'custom' and not (np.diff(param) > 0).all():
                raise ParameterError("Bin limits (param) must be an "
                                     "increasing sequence.")

    @classmethod
    def __create_bins(cls, minValue, maxValue, binType, param):
        """Construct bins."""

        # Create bins
        # left and right are the smallest and largest values
        # that can be placed into the bin.
        if (binType == 'lin' or binType == 'linear'):
            limit_seq = cls.__generateLinbins(minValue, maxValue, param)
            bin_finder = _linBinFinder(limit_seq)
            left, right = minValue, maxValue
            
        elif (binType == 'log' or binType == 'logarithmic'):
            limit_seq = cls.__generateLogbins(minValue, maxValue, param, False)
            # Last bin width will end at the last bin end because the
            # end of the bin is determined freely. Otherwise the last
            # bin would be too small.
            bin_finder = _logBinFinder(limit_seq)
            left, right = minValue, limit_seq[-1]

        elif binType == 'linlog':
            limit_seq = cls.__generateLogbins(minValue, maxValue, param, True)
            bin_finder = _linlogBinFinder(limit_seq)
            left, right = minValue, limit_seq[-1]

        elif binType == 'maxlog' and param is None:
            limit_seq = cls.__generateMaxLogbins(minValue, maxValue)
            bin_finder = _logBinFinder(limit_seq)
            left, right = minValue, maxValue
            
        elif binType == 'maxlog':
            limit_seq = cls.__generateMaxLogbins(minValue, maxValue, param)
            bin_finder = _logBinFinder(limit_seq)
            left, right = minValue, maxValue
            
        elif binType == 'linmaxlog':
            limit_seq = [i-0.5 for i in
                              range(max(0, int(minValue)),
                                    min(12, int(maxValue)+2))]
            if maxValue > limit_seq[-1]:
                limit_seq = limit_seq[:-1]
                tmp = minValue
                minValue = 11
                limit_seq.extend(cls.__generateMaxLogbins(minValue, maxValue, 0.5))
                minValue = tmp
            bin_finder = _linlogBinFinder(limit_seq)
            left, right = minValue, maxValue

        elif binType == 'custom' and (len(param) > 0):
            limit_seq = param
            bin_finder = _binFinder(param)
            left, right = minValue, maxValue
            
        else:
            raise ParameterError("Unidentified parameter combination.")

        return left, right, limit_seq, bin_finder

    @classmethod
    def __inferDataType(cls, minValue, maxValue, dataType):
        """ Infer data type.

        The width of each bin depends on whether the bin may contain
        float or only integers. For example a bin with limits [1.1,
        2.9] has width 1.8 if data is of type 'float' but width 1 if
        data is of type 'int', as only one integer can be placed into
        the bin.

        Unless the data type is explicitely specified, this method is
        used to infer it from the given minimum and maximum values. If
        both minValue and maxValue are integers (type 'int')
        then the data type is also assumed to be integers. Otherwise
        the data is assumed to be floats.
        """
        if dataType == None:
            if (isinstance(minValue, int) and
                isinstance(maxValue, int)):
                return int
            else:
                return float
        else:
            return dataType


    def __new__(cls, dataType, minValue, maxValue, binType, param = None):
        """Initialize bin limits.

        The parameters are identical to those used in Bins.__init__,
        so see that method for explanation.

        If dataType is None, it is inferred from minValue and
        maxValue: if both are of type int, dataType is also int,
        otherwise float.
        """

        # Find data type
        dataType = cls.__inferDataType(minValue, maxValue, dataType)

        # Convert binType to lower case (just in case)
        binType = binType.lower()

        # Make sure the input parameters are valid.
        cls.__check_parameters(minValue, maxValue, binType, param)

        # Get the bin limits
        left, right, limit_seq, bin_finder = cls.__create_bins(
            minValue, maxValue, binType, param)

        # Initialize tuple with the bin limits.
        obj = super(_BinLimits, cls).__new__(cls, limit_seq)

        # Set object variables and return
        obj.minValue, obj.maxValue = minValue, maxValue
        obj.left, obj.right = left, right
        obj.bin_finder = bin_finder
        obj.dataType = dataType
        return obj
        
    def centers(self):
        """Return bin centers as array."""
        bin_centers = []
        for i in range(len(self)-1):
            bin_centers.append(0.5*(self[i+1]+self[i]))
        return np.array(bin_centers)

    def widths(self):
        """Return bin widths as array."""
        if self.dataType == int:
            bin_widths = np.zeros(len(self)-1, int)
            for i in range(len(self)-1):
                bin_widths[i] = ( ceil(min(self[i+1], self.right+0.5))
                                 - ceil(max(self[i], self.left-0.5)) )
        else:
            bin_widths = np.zeros(len(self)-1, float)
            for i in range(len(self)-1):
                bin_widths[i] = float(min(self[i+1], self.right) -
                                      max(self[i], self.left))
        return bin_widths


class Bins(object):
    """ Class for binning 1-dimensional data."""

    def __init__(self, dataType, minValue, maxValue, binType, param = None):
        """Initialize bins.

        Constructs the bins to be used in binning. You can select
        between several different types of bin limits, and if none of
        these seem fit, you can also supply your own bin limits.

        Parameters
        ----------
        dataType : type
            The type of the data, either int or float.  The data type
            affects the bin widths. For example bin [1.1, 2.9] has
            width 1.8 if data is of type 'float' but width 1 if data
            is of type 'int'.
        minValue : float or integer
            The minimum value that can be placed into the bins.
        maxValue : float or integer
            The maximum value that can be placed into the
            bins. maxValue must always be larger than minValue,
            otherwise BinLimitError is raised.
        binType : string
            The method used for constructing bin limits. More
            information below in section 'Bin types'.
        param : float, integer or list
            Additional parameter for controlling the construction of
            bin limits. The meaning of this parameter depends on bin
            type.

        Bin types
        ---------
        There are several methods for constructing the bin limits
        automatically for given parameters. The possible values for
        binType and the corresponding meaning of the extra parameter
        'param' are listed below.
        
        'lin' or 'linear': param = N_bin (integer, > 0)
            Construct N_bin equally sized bins minValue and maxValue, both
            inclusive. The first (last) bin is centered around minValue
            (maxValue).

        'log' or 'logarithmic': param = factor (float, > 1)
            Construct logarithmic bins between positive values minValue
            and maxValue, both inclusive, using factor to increase bin
            size. First bin will be centered at minValue and subsequent
            bins limits are calculated by multiplying previous limit by
            factor.

        'linlog': param = factor (float, > 1)
            Construct linear bins with width 1.0 from max(0, minValue)
            to min(10, maxValue) and logarithmic bins from thereon,
            using factor to increase bin size. Both minValue and
            maxValue must be integers whenever they are smaller than
            11, otherwise expection ValueError is raised.

        'maxlog': param = diff (float, 0 < diff < 1)
            Construct as many logarithmic bins as possible between
            minValue-diff and maxValue+diff so that each bin contains
            at least one integer. minValue and maxValue must be
            integers.

        'linmaxlog': param = diff (float, 0 < diff < 1)
            Same as 'maxlog', but the use linear bins for small values
            exactly as in 'linlog'.

        'custom': param = bin_limits (sequence)
            Uses any arbitrary bin limits. bin_limits must be a
            sequence of growing values. Note that bin[0] is inclusive
            and bin[-1] is exclusive. minValue and maxValue are used
            as the minimum and maximum values when calculating the bin
            width: minValue (maxValue) is the smallest (largest)
            possible value that can be put into the bin. If minValue
            <= bins[0] or maxValue >= bins[-1], the corresponding
            value has no effect.
        """

        if not isinstance(dataType, type):
            raise ParameterError("dataType must be int or float.")
        self.bin_limits = _BinLimits(dataType, minValue, maxValue,
                                     binType, param)
        self.bin_finder = self.bin_limits.bin_finder

    def __len__(self):
        """Return the number of bins."""
        return len(self.bin_limits)-1

    # Create getter for bin centers.
    @property
    def centers(self):
        """Return bin centers as array."""
        try:
            return self._bin_centers
        except AttributeError:
            self._bin_centers = self.bin_limits.centers()
            return self._bin_centers
        
    # Create getter for bin widths.
    @property
    def widths(self):
        """Return bin widths as array."""
        try:
            return self._bin_widths
        except AttributeError:
            self._bin_widths = self.bin_limits.widths()
            return self._bin_widths

    def __check_data_element(self, elem, N):
        """Check one element of input data.

        Makes sure that elem is a sequence and elem[0] fits inside the
        bin limits. The required length N is _not_ checked (for
        performance reasons this is better done when actually needed)
        but is only shown in the error message if elem is not a sequence.
        """
        # Check bin limits and correct sequence type.
        try:
            if (elem[0] < self.bin_limits.minValue or
                elem[0] > self.bin_limits.maxValue):
                raise BinLimitError("Value %g is not in the interval [%g, "
                                    "%g]."% (elem[0],
                                             self.bin_limits.minValue,
                                             self.bin_limits.maxValue))
        except (TypeError, IndexError):
            # TypeError occurs when data is a list and elem is
            # integer or a float. Rather surprisingly, numpy
            # raises an IndexError in the same situation; try for
            # instance creating a=numpy.array([1,2,3]) and then
            # call a[0][0].
            raise DataTypeError("Elements of input data must be sequences"
                                " with length at least %d." % (N,))

    def bin_count(self, coords):
        """
        Bin data and return the number of data points in each bin.

        Parameters
        ----------
        coords : iterable
            The data to be binned. An element coords[j] must be a
            comparable to bin limits, and is placed into the bin i
            with bins[i] <= coords[j] < bins[i+1]

        Returns
        -------
        binned_data : array
            The binned data, with length N, where binned_data[i] is
            the number of data points that fall into the bin.

        Exceptions
        ----------
        Raise BinLimitError if any element does not fit into the bins.
        """
        binCounts = np.zeros(len(self), float)

        for elem in coords:
            # Check element right here because with bin_count it is
            # not required to be a sequence.
            if (elem < self.bin_limits.minValue or
                elem > self.bin_limits.maxValue):
                raise BinLimitError("Value %g is not in the interval [%g, %g]."
                                    % (elem, self.bin_limits.minValue,
                                       self.bin_limits.maxValue) )
            curr_bin = self.bin_finder(elem)
            binCounts[curr_bin] += 1

        return binCounts

    def bin_count_divide(self, coords):
        """
        Bin data and return the number of data points in each bin
        divided by the bin width.

        Parameters
        ----------
        coords : iterable
            The data to be binned. Each element coords[j] must be a
            comparable to bin limits, and is placed into the bin i
            with bins[i] <= coords[j] < bins[i+1]

        Returns
        -------
        binned_data : array
            The binned data, with length N, where binned_data[i] is
            the number of data points that fall into the bin.

        Exceptions
        ----------
        Raise BinLimitError if any element does not fit into the bins.
        """
        return self.bin_count(coords)/self.widths

    def bin_sum(self, data):
        """
        Bin data and return the sum of data points in each bin.

        Parameters
        ----------
        data : iterable
            The data to be binned. Each element must be a pair (coord,
            value). The element is then placed into the bin i with
            bins[i] <= coord < bins[i+1]

        Returns
        -------
        binned_data : ma.masked_array
            The binned data, with length N, where binned_data[i] is
            the sum of all values that fall into the bin. The bins
            with no values are masked. To get a plain list, use
            binned_data.tolist() --- by default the missing values are
            replaced with None.

        Exceptions
        ----------
        Raise BinLimitError if any element does not fit into the bins.
        Raise DataTypeError if data does not consist of pairs. 
        """
        binValues = np.zeros(len(self), float)
        binCounts = np.zeros(len(self), float)

        for elem in data:
            # Make sure the data is valid.
            self.__check_data_element(elem, 2)
            # Find the correct bin and increase count.
            curr_bin = self.bin_finder(elem[0])
            binCounts[curr_bin] += 1
            try:
                binValues[curr_bin] += elem[1]
            except IndexError:
                raise DataTypeError("Elements of input data must be sequences"
                                    " with length at least 2.")

        return np.ma.masked_array(binValues, binCounts == 0)

    def bin_sum_divide(self, data):
        """
        Bin data and return the sum of data points in each bin divided
        by the bin width.

        Parameters
        ----------
        data : iterable
            The data to be binned. Each element must be a pair (coord,
            value). The element is then placed into the bin i with
            bins[i] <= coord < bins[i+1]

        Returns
        -------
        binned_data : ma.masked_array
            The binned data, with length N, where binned_data[i] is
            the sum of all values that fall into the bin, divided by
            the width of the bin. The bins with no values are
            masked. To get a plain list, use binned_data.tolist().

        Exceptions
        ----------
        Raise BinLimitError if any element does not fit into the bins.
        Raise DataTypeError if data does not consist of pairs. 
        """
        return self.bin_sum(data)/self.widths
    
    def bin_average(self, data, variances=False):
        """
        Bin data and return the average of data points in each bin. If
        variances = True, return also the variance in each bin.

        Parameters
        ----------
        data : iterable
            The data to be binned. Each element must be a pair (coord,
            value). The element is then placed into the bin i with
            bins[i] <= coord < bins[i+1]

        Returns
        -------
        variances = False,
            binned_data : ma.masked_array
                The binned data, with length N, where binned_data[i]
                is the average of all values that fall into the
                bin. The bins with no values are masked. To get a
                plain list, use binned_data.tolist().

        variances = True,
            binned_data : tuple (ma.masked_array, ma.masked_array)
                The first element is the average in each bin as
                above, and the second element is the variance of the
                data in each bin.

        Exceptions
        ----------
        Raise BinLimitError if any element does not fit into the bins.
        Raise DataTypeError if data does not consist of pairs. 
        """
        binValues = np.zeros(len(self), float)
        binSquares = np.zeros(len(self), float)
        binCounts = np.zeros(len(self), float)

        for elem in data:
            # Make sure the data is valid.
            self.__check_data_element(elem, 2)
            # Find the correct bin.
            curr_bin = self.bin_finder(elem[0])
            binCounts[curr_bin] += 1
            try:
                binValues[curr_bin] += elem[1]
                binSquares[curr_bin] += elem[1]**2
            except IndexError:
                raise DataTypeError("Elements of input data must be sequences"
                                    " with length at least 2.")

        # Calculate averages as masked array.
        binCounts = np.ma.masked_array(binCounts, binCounts == 0)
        ma_averages = binValues/binCounts
        
        if variances:
            return ma_averages, binSquares/binCounts - ma_averages**2
        else:
            return ma_averages



    def bin_weighted_average(self, data, variances = False):
        """
        Bin data and return the weighted average of data points in
        each bin. If variances = True, return also the weighted
        variance in each bin.

        Parameters
        ----------
        data : iterable
            The data to be binned. Each element must be a triple
            (coord, value, weight). The element is then placed into
            the bin i with bins[i] <= coord < bins[i+1]

        Returns
        -------
        variances = False,
            binned_data : ma.masked_array
                The binned data, with length N, where binned_data[i]
                is the weighted average of all values that fall into
                the bin. The bins with no values are masked. To get a
                plain list, use binned_data.tolist().

        variances = True,
            binned_data : tuple (ma.masked_array, ma.masked_array)
                The first element is the weighted average in each bin
                as above, and the second element is the weighted
                variance of the data in each bin.

        Exceptions
        ----------
        Raise BinLimitError if any element does not fit into the bins.
        Raise DataTypeError if data does not consist of triples. 
        """
        binValues = np.zeros(len(self), float)
        binSquares = np.zeros(len(self), float)
        binCounts = np.zeros(len(self), float)
        binWeights = np.zeros(len(self), float)
        
        for elem in data:
            # Make sure the data is valid.
            self.__check_data_element(elem, 3)
            # Find the correct bin.
            curr_bin = self.bin_finder(elem[0])
            binCounts[curr_bin] += 1
            try:
                binValues[curr_bin] += elem[1]*elem[2]
                binSquares[curr_bin] += elem[1]**2 * elem[2]
                binWeights[curr_bin] += elem[2]
            except IndexError:
                raise DataTypeError("Elements of input data must be sequences"
                                    " with length at least 3.")

        # Calculate weighted average.
        binValues = np.ma.masked_array(binValues, binCounts == 0)
        binWeights[binWeights==0] = 1.0
        ma_wAverages = binValues/binWeights

        if variances:
            return ma_wAverages, binSquares/binWeights - ma_wAverages**2
        else:
            return ma_wAverages

    def bin_median(self, data):
        """
        Bin data and return the median in each bin.

        Parameters
        ----------
        data : iterable
            The data to be binned. Each element must be a pair (coord,
            value). The element is then placed into the bin i with
            bins[i] <= coord < bins[i+1]

        Returns
        -------
        binned_data : ma.masked_array
            The binned data, with length N, where binned_data[i] is
            the median of all values that fall into the bin. The bins
            with no values are masked. To get a plain list, use
            binned_data.tolist().

        Exceptions
        ----------
        Raise BinLimitError if any element does not fit into the bins.
        Raise DataTypeError if data does not consist of pairs. 

        Notes
        -----
        Finding the median of an arbitrary sequence requires first
        saving all elements. If the number of data points is very
        large, this method can take up a large amount of memory.
        """
        binElements = [ [] for i in range(len(self))]

        for elem in data:
            # Make sure the data is valid.
            self.__check_data_element(elem, 2)
            # Find the correct bin.
            curr_bin = self.bin_finder(elem[0])
            # Append the list of elements in the bin. BinLimitError
            # occurs if bin goes over the top, and DataTypeError
            # occurs if elem is not a sequence with length at least 2.
            try:
                binElements[curr_bin].append(elem[1])
            except IndexError:
                if curr_bin > len(self)-1:
                    raise BinLimitError("Value %g is beyond the upper bin "
                                        "limit %g." %
                                        (elem, self.bin_limits.maxValue))
                else:
                    raise DataTypeError("Elements of input data must be "
                                        "sequences with length at least 2.")

        binMedians = np.zeros(len(self), float)

        # Find medians for each bin. If a bin has no values, np.median
        # returns nan.
        for i, elements in enumerate(binElements):
            binMedians[i] = np.median(elements)

        return np.ma.masked_array(binMedians, np.isnan(binMedians))


class Bins2D(object):
    """ Class for binning 2-dimensional data."""

    def __init__(self, X_dataType, X_minValue, X_maxValue, X_binType, X_param,
                 Y_dataType, Y_minValue, Y_maxValue, Y_binType, Y_param):
        """Initialize bins.

        Constructs the bins to be used in binning. You can select
        between several different types of bin limits, and if none of
        these seem fit, you can also supply your own bin limits.

        Usage is identical to Bins, except that you have to supply
        necessary parameters for bins in both x- and y-directions. See
        the documentation of Bins for more information.
        """
        self.x_limits = _BinLimits(X_dataType, X_minValue, X_maxValue,
                                   X_binType, X_param)
        self.y_limits = _BinLimits(Y_dataType, Y_minValue, Y_maxValue,
                                   Y_binType, Y_param)
        self.x_bin_finder = self.x_limits.bin_finder
        self.y_bin_finder = self.y_limits.bin_finder

    @property
    def shape(self):
        """Shape of bins."""
        try:
            return self._shape
        except AttributeError:
            """Return the number of bins in x- and y-directions."""
            self._shape = (len(self.x_limits)-1, len(self.y_limits)-1)
            return self._shape

    # Create getter for bin centers.
    @property
    def centers(self):
        """Return bin centers as 2 arrays (X,Y)."""
        try:
            return self._centers
        except AttributeError:
            self._centers = (self.x_limits.centers(),
                             self.y_limits.centers())
            return self._centers

    @property
    def center_grids(self):
        """Meshgrid of bin centers."""
        try:
            return self._center_grids
        except AttributeError:
            self._center_grids = np.meshgrid(self.x_limits.centers(),
                                             self.y_limits.centers())
            return self._center_grids

    @property
    def edge_grids(self):
        """Meshgrid of bin edges.

        The edge meshgrids should be used with the matplotlib.pcolor
        command.
        """
        try:
            return self._edge_grids
        except AttributeError:
            self._edge_grids = np.meshgrid(self.x_limits,self.y_limits)
            return self._edge_grids


    @property
    def sizes(self):
        """Bin sizes as 2-d array."""
        try:
            return self.bin_widths
        except AttributeError:
            self.bin_widths = np.outer(self.x_limits.widths(),
                                       self.y_limits.widths())
            return self.bin_widths

    def __check_data_element(self, elem, N):
        """Check one element of input data.

        Makes sure that len(elem) >= 2 and the it fits the bin
        limits. The required length of elem is N; this is not checked,
        but only shown in the error message if len(elem) < 2.
        """
        # Check bin limits and correct sequence type.
        try:
            if (elem[0] < self.x_limits.minValue or
                elem[0] > self.x_limits.maxValue):
                raise BinLimitError("X-coordinate %g is not in the interval [%g"
                                    ", %g]." % (elem[0], self.x_limits.minValue,
                                                self.x_limits.maxValue) )
            elif (elem[1] < self.y_limits.minValue or
                  elem[1] > self.y_limits.maxValue):
                raise BinLimitError("Y-coordinate %g is not in the interval [%g"
                                    ", %g]." % (elem[1], self.y_limits.minValue,
                                       self.y_limits.maxValue) )
        except TypeError, IndexError:
            # TypeError occurs when data is a list and elem is
            # integer or a float. Rather surprisingly, numpy
            # raises an IndexError in the same situation; try for
            # instance creating a=numpy.array([1,2,3]) and then
            # call a[0][0].
            raise DataTypeError("Elements of input data must be sequences "
                                "with length at least %d." % (N,))
        
    def bin_count(self, coords):
        """
        Bin data and return the number of data points in each bin.

        Parameters
        ----------
        coords : iterable
            The data to be binned. An element coords[k] must be
            comparable to bin limits, and is placed into the bin (i,j)
            with x_bins[i] <= coords[k][0] < x_bins[i+1]
                 y_bins[j] <= coords[k][1] < x_bins[j+1]

        Returns
        -------
        binned_data : 2-d array
            The binned data, where binned_data[i,j] is the number of
            data points that fall into the bin.

        Exceptions
        ----------
        Raise BinLimitError if any element does not fit into the bins.
        Raise DataTypeError if data does not consist of pairs. 
        """
        binCounts = np.zeros(self.shape, float)

        for elem in coords:
            self.__check_data_element(elem, 2)
            x_bin = self.x_bin_finder(elem[0])
            y_bin = self.y_bin_finder(elem[1])
            binCounts[x_bin, y_bin] += 1

        return binCounts

    def bin_count_divide(self, coords):
        """
        Bin data and return the number of data points in each bin
        divided by the bin width.

        Parameters
        ----------
        coords : iterable
            The data to be binned. An element coords[k] must be
            comparable to bin limits, and is placed into the bin (i,j)
            with x_bins[i] <= coords[k][0] < x_bins[i+1]
                 y_bins[j] <= coords[k][1] < x_bins[j+1]

        Returns
        -------
        binned_data : 2-d array
            The binned data, where binned_data[i,j] is the number of
            data points that fall into the bin divided by the bin
            size.

        Exceptions
        ----------
        Raise BinLimitError if any element does not fit into the bins.
        Raise DataTypeError if data does not consist of pairs. 
        """
        return self.bin_count(coords)/self.sizes

    def bin_sum(self, data):
        """
        Bin data and return the sum of data points in each bin.

        Parameters
        ----------
        data : iterable
            The data to be binned. Each element must be a triple
            (x,y,value), where x and y are comparable to bin
            limits. value is placed into the bin (i,j) with
                x_bins[i] <= x < x_bins[i+1]
                y_bins[j] <= y < y_bins[j+1]

        Returns
        -------
        binned_data : ma.masked_array
            The binned data, with length N, where binned_data[i,j] is
            the sum of all values that fall into the bin. The bins
            with no values are masked. To get a plain list, use
            binned_data.tolist() --- by default the missing values are
            replaced with None.

        Exceptions
        ----------
        Raise BinLimitError if any element does not fit into the bins.
        Raise DataTypeError if data does not consist of triples. 
        """
        binValues = np.zeros(self.shape, float)
        binCounts = np.zeros(self.shape, float)

        for elem in data:
            self.__check_data_element(elem, 3)
            x_bin = self.x_bin_finder(elem[0])
            y_bin = self.y_bin_finder(elem[1])
            binCounts[x_bin, y_bin] += 1
            try:
                binValues[x_bin, y_bin] += elem[2]
            except IndexError:
                raise DataTypeError("Elements of input data must be sequences"
                                    " with length at least 3.")

        return np.ma.masked_array(binValues, binCounts == 0)

    def bin_sum_divide(self, data):
        """
        Bin data and return the sum of data points in each bin divided
        by the bin width.

        Parameters
        ----------
        data : iterable
            The data to be binned. Each element must be a triple
            (x,y,value), where x and y are comparable to bin
            limits. value is placed into the bin (i,j) with
                x_bins[i] <= x < x_bins[i+1]
                y_bins[j] <= y < y_bins[j+1]

        Returns
        -------
        binned_data : ma.masked_array
            The binned data, with length N, where binned_data[i,j] is
            the sum of all values that fall into the bin, divided by
            the bin size. The bins with no values are masked. To get a
            plain list, use binned_data.tolist() --- by default the
            missing values are replaced with None.

        Exceptions
        ----------
        Raise BinLimitError if any element does not fit into the bins.
        Raise DataTypeError if data does not consist of triples. 
        """
        return self.bin_sum(data)/self.sizes
    
    def bin_average(self, data, variances = False):
        """
        Bin data and return the average of data points in each bin. If
        variances = True, return also the variance in each bin.

        Parameters
        ----------
        data : iterable
            The data to be binned. Each element must be a triple
            (x,y,value), where x and y are comparable to bin
            limits. value is placed into the bin (i,j) with
                x_bins[i] <= x < x_bins[i+1]
                y_bins[j] <= y < y_bins[j+1]

        Returns
        -------
        variances = False,
            binned_data : ma.masked_array
                The binned data, with length N, where binned_data[i,j]
                is the average of all values that fall into the
                bin. The bins with no values are masked. To get a
                plain list, use binned_data.tolist() --- by default
                the missing values are replaced with None.

        variances = True,
            binned_data : tuple (ma.masked_array, ma.masked_array)
                The first element is the average in each bin as
                above, and the second element is the variance of the
                data in each bin.

        Exceptions
        ----------
        Raise BinLimitError if any element does not fit into the bins.
        Raise DataTypeError if data does not consist of triples. 
        """
        
        binValues = np.zeros(self.shape, float)
        binSquares = np.zeros(self.shape, float)
        binCounts = np.zeros(self.shape, float)

        for elem in data:
            self.__check_data_element(elem, 3)
            x_bin = self.x_bin_finder(elem[0])
            y_bin = self.y_bin_finder(elem[1])
            binCounts[x_bin, y_bin] += 1
            try:
                binValues[x_bin, y_bin] += elem[2]
                binSquares[x_bin, y_bin] += elem[2]**2
            except IndexError:
                raise DataTypeError("Elements of input data must be sequences"
                                    " with length at least 3.")

        # Calculate averages as masked arrays.
        binCounts = np.ma.masked_array(binCounts, binCounts == 0)
        ma_averages = binValues/binCounts

        if variances:
            return ma_averages, binSquares/binCounts - ma_averages**2
        else:
            return ma_averages

    def bin_weighted_average(self, data, variances = False):
        """
        Bin data and return the weighted average of data points in
        each bin. If variances = True, return also the weighted
        variance in each bin.

        Parameters
        ----------
        data : iterable
            The data to be binned. Each element must be a tuple
            (x,y,value,weight), where x and y are comparable to bin
            limits. value is placed into the bin (i,j) with
                x_bins[i] <= x < x_bins[i+1]
                y_bins[j] <= y < y_bins[j+1]

        Returns
        -------
        variances = False,
            binned_data : ma.masked_array
                The binned data, with length N, where binned_data[i,j]
                is the weighted average of all values that fall into
                the bin. The bins with no values are masked. To get a
                plain list, use binned_data.tolist() --- by default
                the missing values are replaced with None.

        variances = True,
            binned_data : tuple (ma.masked_array, ma.masked_array)
                The first element is the weighted average in each bin
                as above, and the second element is the weighted
                variance of the data in each bin.

        Exceptions
        ----------
        Raise BinLimitError if any element does not fit into the bins.
        Raise DataTypeError if data does not consist of triples. 
        """
        binValues = np.zeros(self.shape, float)
        binSquares = np.zeros(self.shape, float)
        binCounts = np.zeros(self.shape, float)
        binWeights = np.zeros(self.shape, float)
        
        for elem in data:
            self.__check_data_element(elem, 4)
            x_bin = self.x_bin_finder(elem[0])
            y_bin = self.y_bin_finder(elem[1])
            binCounts[x_bin, y_bin] += 1
            try:
                binValues[x_bin, y_bin] += elem[2]*elem[3]
                binSquares[x_bin, y_bin] += elem[2]**2 * elem[3]
                binWeights[x_bin, y_bin] += elem[3]
            except IndexError:
                raise DataTypeError("Elements of input data must be sequences"
                                    " with length at least 4.")

        # Calculate weighted average.
        binValues = np.ma.masked_array(binValues, binCounts == 0)
        binWeights[binWeights==0] = 1.0
        ma_wAverages = binValues/binWeights

        if variances:
            return ma_wAverages, binSquares/binWeights - ma_wAverages**2
        else:
            return ma_wAverages

    def bin_median(self, data):
        """
        Bin data and return the median in each bin.

        Parameters
        ----------
        data : iterable
            The data to be binned. Each element must be a triple
            (x,y,value), where x and y are comparable to bin
            limits. value is placed into the bin (i,j) with
                x_bins[i] <= x < x_bins[i+1]
                y_bins[j] <= y < y_bins[j+1]

        Returns
        -------
        binned_data : ma.masked_array
            The binned data, with length N, where binned_data[i,j] is
            the median of all values that fall into the bin. The bins
            with no values are masked. To get a plain list, use
            binned_data.tolist() --- by default the missing values are
            replaced with None.

        Exceptions
        ----------
        Raise BinLimitError if any element does not fit into the bins.
        Raise DataTypeError if data does not consist of triples. 

        Notes
        -----
        Finding the median of an arbitrary sequence requires first
        saving all elements. If the number of data points is very
        large, this method can take up a large amount of memory.
        """
        
        binElements = [ [ [] for i in range(self.shape[1])]
                        for j in range(self.shape[0])]

        for elem in data:
            self.__check_data_element(elem, 3)
            x_bin = self.x_bin_finder(elem[0])
            y_bin = self.y_bin_finder(elem[1])
            try:
                binElements[x_bin][y_bin].append(elem[2])
            except IndexError:
                raise DataTypeError("Elements of input data must be sequences"
                                    " with length at least 3.")

        binMedians = np.zeros(self.shape, float)

        # Find medians for each bin. If a bin has no values, np.median
        # returns nan.
        for i, i_elements in enumerate(binElements):
            for j, j_elements in enumerate(i_elements):
                binMedians[i,j] = np.median(j_elements)

        return np.ma.masked_array(binMedians, np.isnan(binMedians))


if __name__ == '__main__':
    """Run unit tests if called."""
    from tests.test_binner import *
    unittest.main()
