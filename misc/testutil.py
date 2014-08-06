
import warnings

import networkx

class NotTestedWarning(Warning):
    """Emit a warning for untested code.

    Consider using the warn_untested function instead.

    Sample usage::
        import verkko.misc.testutil as testutil
        import warnings
        warnings.warn('untested', testutil.NotTestedWarning)
    """
    pass
def warn_untested(msg="This code is untested"):
    """Emit a warning for untested code.

    To use this, simply call this in an untested function.  When the
    function runs, it will emit a warning like this::

        verkko/graph/nxutil.py:63: NotTestedWarning: This code is untested
          import verkko.misc.testutil as testutil ; testutil.warn_untested()


    Arguments:
        message: str, warning message, default 'This code is untested'
            This is printed as the warning.  You could include extra
            information here.

    """
    warnings.warn(msg, NotTestedWarning, stacklevel=2)


def assert_isomorphic(g1, g2, msg=None):
    """Assertion function for networkx isomorphism"""
    if not networkx.is_isomorphic(g1, g2):
        msg_ = ["%r and %r not isomorphic"%(g1, g2),
                #"A: %s"%networkx.to_dict_of_lists(g1),
                #"B: %s"%networkx.to_dict_of_lists(g2),
                ]
        n1 = set(g1.nodes_iter())
        n2 = set(g2.nodes_iter())
        if n1 != n2:
            msg_.append("Nodes in 1 only: %s"%(n1-n2))
            msg_.append("Nodes in 2 only: %s"%(n2-n1))
        e1 = set(frozenset((a,b)) for a,b in g1.edges_iter())
        e2 = set(frozenset((a,b)) for a,b in g2.edges_iter())
        if e1 != e2:
            msg_.append("Edges in 1 only: %s"%(' '.join('(%s,%s)'%(a,b) for a,b in e1-e2)))
            msg_.append("Edges in 2 only: %s"%(' '.join('(%s,%s)'%(a,b) for a,b in e2-e1)))
        if msg: msg_.insert(0, msg)
        raise AssertionError('\n'.join(msg_))

