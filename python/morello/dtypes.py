import cython
import numpy as np


@cython.dataclasses.dataclass(frozen=True)
@cython.cclass
class Dtype:
    size: int  # in bytes
    c_type: str
    int_fmt_macro: str
    np_type: np.dtype
    short_name: str

    @staticmethod
    def from_short_name(short_name: str) -> "Dtype":
        for dtype in ALL_DTYPES:
            if dtype.short_name == short_name:
                return dtype
        raise ValueError(f"Unknown short name: {short_name}")

    def __hash__(self) -> int:
        return hash(self.short_name)

    def __reduce__(self):
        return Dtype.from_short_name, (self.short_name,)

    def __str__(self):
        return self.short_name


Uint8 = Dtype(
    size=1, c_type="uint8_t", int_fmt_macro="PRIu8", np_type=np.uint8, short_name="u8"
)
Uint32 = Dtype(
    size=4,
    c_type="uint32_t",
    int_fmt_macro="PRIu32",
    np_type=np.uint32,
    short_name="u32",
)

ALL_DTYPES = [Uint8, Uint32]
