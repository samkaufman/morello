
from morello import layouts


def test_dimdroplayout_normalization_of_row_major_1():
    rm2 = layouts.row_major(2) # 0, 1
    assert layouts.DimDropLayout(rm2, dropped_dims=frozenset({0})).normalize() \
        == layouts.row_major(1)

def test_dimdroplayout_normalization_of_row_major_2():
    rm2 = layouts.row_major(2) # 0, 1
    assert layouts.DimDropLayout(rm2, dropped_dims=frozenset({1})).normalize() \
        == layouts.row_major(1)

def test_dimdroplayout_normalization_of_row_major_3():
    rm3 = layouts.row_major(3) # 0, 1, 2
    assert layouts.DimDropLayout(rm3, dropped_dims=frozenset({1})).normalize() \
        == layouts.row_major(2)

def test_dimdroplayout_normalization_of_packedlayout_1():
    p_a = layouts.PackedLayout(dim_count=4, strip_dim=1, strip_size=4)
    assert layouts.DimDropLayout(p_a, dropped_dims=frozenset({2, 3})).normalize() \
        == layouts.row_major(2)

def test_dimdroplayout_normalization_of_packedlayout_2():
    p_b = layouts.PackedLayout(dim_count=4, strip_dim=1, strip_size=4)
    assert layouts.DimDropLayout(p_b, dropped_dims=frozenset({1})).normalize() \
        == layouts.row_major(3)

def test_dimdroplayout_normalization_of_packedlayout_3():
    p_c = layouts.PackedLayout(dim_count=5, strip_dim=1, strip_size=4)
    d_p_c = layouts.DimDropLayout(p_c, dropped_dims=frozenset({2, 3})).normalize()
    assert d_p_c.normalize() == d_p_c

def test_transposelayout_normalization_of_row_major_1():
    rm2 = layouts.row_major(2) # 0, 1
    assert layouts.TransposeLayout(rm2, swap_dims=(0, 1)).normalize() \
        == layouts.StandardLayout((1, 0))

def test_transposelayout_normalization_of_row_major_2():
    rm3 = layouts.row_major(3) # 0, 1, 2
    assert layouts.TransposeLayout(rm3, swap_dims=(0, 1)).normalize() \
        == layouts.StandardLayout((1, 0, 2))

def test_transposelayout_normalization_of_row_major_3():
    rm3 = layouts.row_major(3) # 0, 1, 2
    assert layouts.TransposeLayout(rm3, swap_dims=(1, 2)).normalize() \
        == layouts.StandardLayout((0, 2, 1))

def test_transposelayout_normalization_of_row_major_4():
    rm3 = layouts.row_major(3) # 0, 1, 2
    assert layouts.TransposeLayout(rm3, swap_dims=(0, 2)).normalize() \
        == layouts.StandardLayout((2, 1, 0))
