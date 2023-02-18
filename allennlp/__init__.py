# We get a lot of these spurious warnings,
# see https://github.com/ContinuumIO/anaconda-issues/issues/6678
import warnings  # noqa

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

try:
    # On some systems this prevents the dreaded
    # ImportError: dlopen: cannot load any more object with static TLS
    import torch, numpy  # noqa

except ModuleNotFoundError:
    print(
        "We require the python packages Pytorch and Numpy to be installed."
    )
    raise

from allennlp.version import VERSION as __version__  # noqa
