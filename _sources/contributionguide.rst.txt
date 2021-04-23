##################
Contribution Guide
##################

***************
1. Introduction
***************

This contribution guide focuses on the code style. Together with *PEP8*
(van Rossum et al., 2001) and the *Guide of NumPy/SciPy Documentation*
(Gommers et al, 2017) it describes the conventions to
follow while designing the software viroconweb. The guide is supposed to
help the developer to stick to a constant style and layout, which will
make the code more readable. This style guide provides conventions and
should not be seen as a dogma. If it serves the readability you can
deviate from the given style.

**************
2. Code Layout
**************

2.1 Indentation
===============

Use four spaces to differ between two indention levels. Do not use tabs
unless you need to be conform with code written by some other developer
(Rossum, Warsaw, Coghlan, 2001).

2.2 Line length
===============

The maximum length of a line should be 79 chars. Long lines of code can
be broken up by a backslash. Further details are described in PEP8:
https://www.python.org/dev/peps/pep-0008/#maximum-line-length

2.3 Blank Lines
===============

Use two blank lines before you define a new class and one blank line
before method definitions. Blank lines can border a group of two
functions which are related. You can use them to differ between logic
paragraphs (Rossum, Warsaw, Coghlan, 2001).

2.4 Imports
===========

Imports are located at the top of the document right behind docstrings
and method comments. Use separate lines for every import (Rossum,
Warsaw, Coghlan, 2001).

***********
3. Comments
***********

3.1 Block Comments
==================

Block comments can describe either the whole following code or just a
part of it. Every line starts with # followed by one space (Rossum,
Warsaw, Coghlan, 2001).

3.2 Short Comments
==================

Short comments can just be single words. They can be located between
statements.

3.3 Docstrings
==============

Functions, modules or classes can be extended by docstrings (Rossum,
Warsaw, Coghlan, 2001). Like block comments, docstrings can be written
over multiple lines. To differ between several sections write a header
followed by a underline which is as long as the header. To end a section
use two blank lines. To distinguish a section particularly use
indentions. There are different sections of docstrings that can be used:

3.3.1. Short Summary
--------------------

Describes what is happening in just one line. Can also give information
about returns (Gommers, 2017), e.g. Find biggest elements to sum to
reach limit.

3.3.2. Extended Summary
-----------------------

Describe what happens in the section using whole sentences. You can use
the parameters and function name to make sure the functionality is clear
(Gommers, 2017), e.g. Sorts array, and calculates the cumulative sum.
Returns a boolean array with the same shape as array indicating the
fields summed to reach limit, as well as the last value added.

3.3.3. Parameters
-----------------

To describe Parameters put variables into single back ticks. Surround
the colon with single spaces or leave it when the type is not given.
Always be accurate describing Parameters (Gommers, 2017), e.g.

.. code:: python

    """
    Parameters
    ——————–
    x : type
        Description of parameter `x`.
    y
        Description of parameter `y` (with type not specified)

3.3.4. Returns
--------------

Like Parameters but every typ of the return needs to be mentioned
(Gommers, 2017), e.g.

.. code:: python

    """
    Returns
    ————–
    summed_fields : ndarray, dtype=Bool
        Boolean array of shape like  with True if element was used in summation.
    last_summed : float
        Element that was added last to the sum.
    """

3.3.5. Raises
-------------

Errors that may appear (Gommers, 2017), e.g.

.. code:: python

    """
    Raises
    ———–
    ValueError
        If ‘array‘ contains nan.
    """

3.3.6. Notes
------------

In this section you can write extra information to the code. This may be
critical statements or just comments (Gommers, 2017), e.g.

.. code:: python

    """
    Notes
    ———
    The following attributes/methods need to be initialised by child classes:
        - name
        - _scipy_cdf
        - _scipy_i_cdf
    """

********
4. Tests
********

Make sure that after your contribution all tests still run successfully.
To run the tests type

.. code:: console

    pytest

If you implement a new feature, write a new test, which
covers the new feature.

**********
References
**********

R. Gommers, endolith, chebee7i, T. Kluyver, P. de Buyl, C. Harris et al. (2017):
A Guide to NumPy/SciPy Documentation.
https://github.com/numpy/numpy/blob/maintenance/1.14.x/doc/HOWTO_DOCUMENT.rst.txt
(last access: July 9th 2018)

van Rossum, G.; Warsaw, B.; Coghlan, N. (2001): Style Guide for Python
Code.https://www.python.org/dev/peps/pep-0008 (last access
May 18th 2018)
