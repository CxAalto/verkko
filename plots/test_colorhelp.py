import colorhelp
from matplotlib.colors import ColorConverter


def test():
    n = 1
    colors = colorhelp.get_distinct_colors(n)
    assert len(colors) is 1
    n = 10
    colors = colorhelp.get_distinct_colors(n)
    assert len(colors) is 10
    cc = ColorConverter()
    for color in colors:
        assert len(cc.to_rgba(color)) is 4
