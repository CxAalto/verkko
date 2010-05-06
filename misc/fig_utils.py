# Different command for making figures.

import sys
import os
import pylab
import matplotlib.ticker as ticker
import numpy as np

def savefig(fig, name, extensions=None, verbose=False):
    """Save figure.

    Save matplotlib.figure object `fig` as `name`.EXT, where EXT are
    given in `extensions`. If only one save type is used, the full
    name (including the extension) can also be given as `name`.

    Note! Saving as 'svg' requires that the program 'pdf2svg' is
    installed on your system.
    """

    if len(name) == 0:
        raise ValueError("File name can not be empty.")

    if extensions is None:
        fields = name.split(".")
        if len(fields) == 1:
            raise ValueError("File name must contain an extension if"
                             " extensions are not given explicitely.")
        extensions = fields[-1]
        name = ".".join(fields[:-1])

    if isinstance(extensions, str):
        extensions = (extensions,)
    
    # Check if pdf should be generated (both eps and svg will be
    # created from pdf) and generate if necessary.
    pdf_generated = False
    pdf_tmp = "%s_tmp_%d.pdf" % (name, np.random.randint(100000))
    if set(['pdf','svg','eps']).intersection(extensions):
        fig.savefig(pdf_tmp)
        pdf_generated = True

    for ext in extensions:
        if not isinstance(ext, str):
            raise ValueError("'extensions' must be a list of strings.")
        if ext[0] == '.':
            ext = ext[1:]
            
        if ext == 'eps':
            pipe = os.popen("pdftops -eps %s %s.eps" % (pdf_tmp, name))
            exit_status = pipe.close()
            if exit_status:
                if os.WEXITSTATUS(exit_status) == 127:
                    sys.stderr.write("%s could not be created because program "
                                     "'pdftoeps' could not be found.\n" % (ext,))
                else:
                    sys.stderr.write("Problem saving '%s'.\n" % (ext,))
        elif ext == 'svg':
            pipe = os.popen("pdf2svg %s %s.svg" % (pdf_tmp, name))
            exit_status = pipe.close()
            if exit_status:
                if os.WEXITSTATUS(exit_status) == 127:
                    sys.stderr.write("%s could not be created because program "
                                     "'pdf2svg' could not be found.\n" % (ext,))
                else:
                    sys.stderr.write("Problem saving '%s'.\n" % (ext,))
        elif ext != 'pdf':
            # fig.savefig raises a ValueError if the extension is not identified.
            fig.savefig(name + "." + ext)

    if pdf_generated:
        if 'pdf' in extensions:
            os.popen("mv %s %s.pdf" % (pdf_tmp,name))
        else:
            os.popen("rm %s" % (pdf_tmp,))

def get_rcParams(fig_width_cm, fig_ratio = 0.8, font_sizes = None):
    """Set good parameters for LaTeX-figures.

    The idea idea is to set the figure width in centimeters to be the
    same as the final size in your LaTeX document. This way the font
    sizes will be correct also.

    Parameters
    ----------
    fig_width_cm: int or float
        The width of the final figure in centimeters.
    fig_ratio: float (between 0 and 1)
        The ratio height/width. < 1 is landscape, 1.0 is square and
        > 1.0 is portrait.
    font_sizes: dictionary
        The font sizes used in the figure. Default is size 8 for
        everything else expect 10 for the title. Possible keys are
        'default', 'label', 'title', 'text', 'legend' and 'tick'.
        'default' is used when the specific value is not defined,
        other keys should be self explanatory.
    """
    default_font_sizes = {'label':8, 'title':10, 'text':8, 'legend':8, 'tick':8}
    font_sizes = (font_sizes or {})
    for k in default_font_sizes:
        if k not in font_sizes:
            font_sizes[k] = (font_sizes.get('default') or default_font_sizes[k])

    inches_per_cm = 1/2.54
    fig_width = 1.0*fig_width_cm*inches_per_cm  # width in inches
    fig_height = 1.0*fig_width*fig_ratio        # height in inches
    fig_size =  [fig_width,fig_height]
    params = {'font.family':'serif',
              'font.serif':'Computer Modern Roman',
              'axes.labelsize': font_sizes['label'],
              'axes.titlesize': font_sizes['title'],
              'text.fontsize': font_sizes['text'],
              'font.size': font_sizes['text'],
              'legend.fontsize': font_sizes['legend'],
              'xtick.labelsize': font_sizes['tick'],
              'ytick.labelsize': font_sizes['tick'],
              'text.usetex': True,
              'figure.figsize': fig_size,
              'legend.labelspacing': 0.0,
              'lines.markersize': 3,
              'lines.linewidth': 0.5}
    return params

class EvenExpFormatter(ticker.LogFormatterMathtext):
    """Print labels only for even exponentials. Exponents given in
    'exclude' will also be skipped.
    """
    def __init__(self, base=10.0, labelOnlyBase=True, exclude=None):
        if exclude == None:
            self.exclude = []
        else:
            self.exclude = exclude
        ticker.LogFormatterMathtext.__init__(self, base, labelOnlyBase)
    
    def __call__(self, val, pos=None):
        fx = int(np.floor(np.log(abs(val))/np.log(self._base) +0.5)) 
        isDecade = self.is_decade(fx)
        if not isDecade and self.labelOnlyBase:
            return ''
        if (fx%2)==1 or (fx in self.exclude): # odd, skip
            return ''

        return ticker.LogFormatterMathtext.__call__(self, val, pos)

class SkipMathFormatter(ticker.LogFormatterMathtext):
    """Skip exponents given in 'exclude'.
    """
    def __init__(self, base=10.0, labelOnlyBase=True, exclude=None):
        if exclude == None:
            self.exclude = []
        else:
            self.exclude = exclude
        ticker.LogFormatterMathtext.__init__(self, base, labelOnlyBase)
    
    def __call__(self, val, pos=None):
        fx = int(np.floor(np.log(abs(val))/np.log(self._base) +0.5)) 
        isDecade = self.is_decade(fx)
        if not isDecade and self.labelOnlyBase:
            return ''
        if fx in self.exclude: # skip
            return ''

        return ticker.LogFormatterMathtext.__call__(self, val, pos)

class SelectiveScalarFormatter(ticker.ScalarFormatter):
    """Print only the ticks in the input list.
    """
    def __init__(self, printList=None, useOffset=True, useMathText=False):
        if printList is None:
            printList = []
        self.printList = printList
        ticker.ScalarFormatter.__init__(self, useOffset, useMathText)
    
    def __call__(self, val, pos=None):
        if val in self.printList:
            return ticker.ScalarFormatter.__call__(self, val, pos)
        else:
            return ''


class DivisorFormatter(ticker.FormatStrFormatter):
    """Divide all numbers by a constant.
    """
    def __init__(self, fmt, divisor=None):
        if divisor == None:
            self.divisor = 1
        else:
            self.divisor = divisor
        ticker.FormatStrFormatter.__init__(self, fmt)
    
    def __call__(self, val, pos=None):
        return ticker.FormatStrFormatter.__call__(self, int(1.0*val/self.divisor), pos)


