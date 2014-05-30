VERKKOLIB (name under progress)
-------------------------------

The group library for general purpose analysis and visualization code.
This repository should not contain project-specific code.
Most of the code in this repo is written in Python, but also other languages are
allowed.
Use common sense when combining different programming languages.


Dependencies:
-------------

Structure of the repository:
----------------------------

* plots:
    - All the functionality related to plotting and visualization.

* misc:
    - All your random stuff.

* other names:
    - Added as needed for sub-projects.

* u:
    - Personal general purpose code, without testing or
    backwards-compatability requirements.  Used for things too
    specific to be worth refining, but should be somewhere.
    Contains separate subdirectories for different group members.
    e.g., u/darst.


About code backwards-comptibilty and versioning
-----------------------------------------------

The idea of backwards compatibility is that, once written, code should
continue to work the same in the future even if more features are
added. For a big shared library like this, this is a very important
(and difficult) topic.

1. Code relating to scientific calculations should absolutely remain
backwards compatible.  For example, changing a default option may
affect the integreity of other people's work.

2. All other code should remain as backwards-compatible as possible,
for example plotting related code.  Small changes, such as
improvements to plot output formats, are generally OK but use your
judgement.

3. If need for "rewriting" a function exists, the new version number
should be added to the end of the name:

| Old:      | New:          |
| --------- | ------------- |
| func      | func2         |
| module.py | module2.py    |

At this point, you can take the opportunity to change anything and
make general improvements. This can be done at both the function and
module level.  Alternatively, you could take the opportunity to change
the name to something else that is more descriptive.

4. These rules apply to mature code, not things that are too new to be
used by many people.  It's always best to design things properly in
the first place, so take the time to talk to others and adjust things
while they are new in order to design things as well as possible at
the beginning.


CODING STYLE
--------------
In general, we try to stick to the PEP style and documentation
guidelines:
  http://legacy.python.org/dev/peps/pep-0008/
  http://legacy.python.org/dev/peps/pep-0020/
  http://legacy.python.org/dev/peps/pep-0257/

For a brief summary, see the file python_style.md of this repository.


CODE TESTING PRACTICES
-----------------------

All scientific code should be tested.  However, reality indicates that
tests slowly evolve.

When submitting code, also test cases covering every line of the code are required (?).

However, for the following directories (only one so far):

* u/

no testing is required.
(For u/ this condition is relaxed as the code there is personal and/or
under development)


With visualizations/plotting routines, it is not required to provide explicit 
tests. (testing )
However, an example test/documentation script is required.


PEER REVIEWING CODE(?)
----------------------
Too time-consuming (?)


But at least improve on other people's code when you use it!

*  When you use some code, and find something badly documented/tested etc. 
please improve the code!
*  Don't be too afraid of 'breaking things', everything can be restored!

