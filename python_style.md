# Python style and documentation guide for the group code-library

We try to follow the PEP [style](http://legacy.python.org/dev/peps/pep-0008/), 
and [docstring](http://legacy.python.org/dev/peps/pep-0257/) guidelines, of which the following are modified excerpts.

The reason for aiming to proper style and documentation is the following:

> Code is read much more often than it is written.
> Thus readability counts.


### Use 4 spaces!
Python 3 does not even allow mixed use of tabs and spaces)

### Limit lines to 79 characters
- Possible to have multiple source code windows open at the same time
- Easier comparison with code difference tools


### Imports
Put imports on separate lines:
* Yes

```python
import sys
import os
```

* No

```python
import sys, os
```

2. Wildcards imports are **NOT** preferred:
	* However, from `pylab import *` might not be always that dangerous.


### Comments
Please comment your code reasonably.

### Naming conventions
Modules:
	- Short lowercase, like `plots.py`
Class names:
	- CapWords, such as `MyClass`
Function names:
	- lowercase words separated with underscores, e.g., `my_great_func()`

Constants (on the module level):
	- All capital letters with underscores, e.g., "VALUE_OF_PI"

### Documenting code

#### Docstrings:

Use one-liners for really obvious cases:

```python
	def return(x, y):
		""" Return the sum of x and y."""
		return x + y
```

Multiline docstrings:
	* First one line summary, then more elaborate description, if necessary.

```python
def complex(real=0.0, imag=0.0):
    """Form a complex number.

    Keyword arguments:
    real -- the real part (default 0.0)
    imag -- the imaginary part (default 0.0)

    Returns:
    complex -- a complex number
    """
    if imag == 0.0 and real == 0.0:
        return complex_zero
```
