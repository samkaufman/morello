from morello import layouts


def test_dim_drop_of_row_major_1():
    rm2 = layouts.row_major(2)  # 0, 1
    assert rm2.dim_drop({0}, 1) == (layouts.row_major(1), 1)


def test_dim_drop_of_row_major_2():
    rm2 = layouts.row_major(2)  # 0, 1
    assert rm2.dim_drop({1}, 1) == (layouts.row_major(1), 0)


def test_dim_drop_of_row_major_3a():
    rm3 = layouts.row_major(3)  # 0, 1, 2
    assert rm3.dim_drop({1}, 2) == (layouts.row_major(2), 1)


def test_dim_drop_of_row_major_3b():
    rm3 = layouts.row_major(3)  # 0, 1, 2
    assert rm3.dim_drop({1}, 1) == (layouts.row_major(2), 1)


def test_dim_drop_of_packedlayout_1():
    p_a = layouts.PackedLayout(dim_count=4, strip_dim=1, strip_size=4)
    assert p_a.dim_drop({2, 3}, 2) == (layouts.row_major(2), 0)


def test_dim_drop_of_packedlayout_1b():
    p_a = layouts.PackedLayout(dim_count=4, strip_dim=1, strip_size=4)
    assert p_a.dim_drop({0, 2, 3}, 2) == (layouts.row_major(1), 0)


def test_dim_drop_of_packedlayout_1c():
    p_a = layouts.PackedLayout(dim_count=4, strip_dim=1, strip_size=4)
    # Resulting contiguous should be zero, not 1, because, while the innermost
    # physical (packed) dimension is contiguous, its corresponding outer/block
    # dimension is not.
    assert p_a.dim_drop({0, 2, 3}, 3) == (layouts.row_major(1), 0)


def test_dim_drop_of_packedlayout_1d():
    p_a = layouts.PackedLayout(dim_count=4, strip_dim=1, strip_size=4)
    assert p_a.dim_drop({0, 2, 3}, 4) == (layouts.row_major(1), 1)


def test_dim_drop_of_packedlayout_1e():
    p_a = layouts.PackedLayout(dim_count=4, strip_dim=1, strip_size=4)
    e = (layouts.PackedLayout(dim_count=3, strip_dim=1, strip_size=4), 1)
    assert p_a.dim_drop({3}, 1) == e


def test_dim_drop_of_packedlayout_2():
    p_b = layouts.PackedLayout(dim_count=4, strip_dim=1, strip_size=4)
    assert p_b.dim_drop({1}, 1) == (layouts.row_major(3), 0)


def test_dim_drop_of_packedlayout_3():
    p_c = layouts.PackedLayout(dim_count=5, strip_dim=1, strip_size=4)
    d_p_c = p_c.dim_drop({2, 3}, 6)
    assert d_p_c == (layouts.PackedLayout(dim_count=3, strip_dim=1, strip_size=4), 4)


# def test_dim_drop_of_packedlayout_4():
#     p_c = layouts.PackedLayout(dim_count=4, strip_dim=1, strip_size=4)
#     d_p_c = p_c.dim_drop({2, 3}, 6)
#     assert (p_c, 4) == d_p_c


def test_transpose_normalization_of_row_major_1():
    rm2 = layouts.row_major(2)  # 0, 1
    assert rm2.transpose((0, 1), 2) == (layouts.StandardLayout((1, 0)), 2)


def test_transpose_normalization_of_row_major_2():
    rm3 = layouts.row_major(3)  # 0, 1, 2
    assert rm3.transpose((0, 1), 2) == (layouts.StandardLayout((1, 0, 2)), 2)


def test_transpose_normalization_of_row_major_3():
    rm3 = layouts.row_major(3)  # 0, 1, 2
    assert rm3.transpose((1, 2), 1) == (layouts.StandardLayout((0, 2, 1)), 1)


def test_transpose_normalization_of_row_major_4():
    rm3 = layouts.row_major(3)  # 0, 1, 2
    assert rm3.transpose((0, 2), 0) == (layouts.StandardLayout((2, 1, 0)), 0)
