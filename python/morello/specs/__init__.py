"""specs.py contains the Spec language.

In Morello, a program's algorithm is independent of its schedule. A schedule's
logical semantics are described by a Spec.
"""

from .base import Spec
from .compose import Compose
from .conv import Convolution, ConvolutionAccum
from .matmul import Matmul, MatmulAccum
from .moves import Load, Store
from .original import *
from .reducesum import ReduceSum, ReduceSumAccum
from .tensorspec import LayoutDoesntApplyError, OversizedVectorError, TensorSpec
from .zero import Zero
