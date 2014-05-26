VERKKOLIB (name under progress)
-------------------

The group library for general purpose analysis and visualization code.
This repository should not contain project-specific code.
Most of the code in this repo is written in Python, but also other languages are allowed.
Use common sense when combining languages of different.


Dependencies:
-------------

Structure of the repository:
-----------

plots:
    All the functionality related to plotting and visualization.

misc:
    All your random stuff.

u:
    Personal general purpose code, without testing requirements. 
    Contains separate subdirectories for different group members.
    e.g., u/darst.


About code versioning
--------------------

1. The underlying idea is that all code in the main directories should be backwards compatible. 
2. If need for "rewriting" a function exists, the new version number should be added to the end of the name:

| Old:     | New:          |
| -------- | ------------- |
| verkko   | verkko_v2.py  |
| func     | func_v2.py    |

 


CODING STYLE
--------------
In general, we try to stick to the PEP style and documentation guidelines:

For a brief summary, see the file python_style.md of this repository.


CODE TESTING PRACTICES
-----------------------
All code should be tested before submitting, except for the following directories:
u/

When submitting code, also test cases covering every line of the code are required (?).

With visualizations/plotting routines, it is not required to provide explicit tests.
However, also with them a test/documentation script is necessary.


PEER REVIEWING CODE(?)
----------------------



