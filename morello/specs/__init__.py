"""specs.py contains the Spec language.

In Morello, a program's algorithm is independent of its schedule. A schedule's
logical semantics are described by a Spec.
"""

from .base import Spec
from .compose import Compose
from .conv import Convolution
from .matmul import Matmul
from .moves import Load, Store
from .original import *
from .reducesum import ReduceSum
from .tensorspec import HvxVmemTensorSpec, TensorSpec
from .zero import Zero