macro_rules! define_x86_vec_types {
    ($name:ident, $len:expr $(, $extra:expr )* $(,)?) => {
        const $name: [VecType; $len] = [
            VecType {
                dtype: Dtype::Bfloat16,
                value_cnt: 16,
                name: "vbf16_16",
                native_type_name: "__m256i",
            },
            VecType {
                dtype: Dtype::Bfloat16,
                value_cnt: 8,
                name: "vbf16_8",
                native_type_name: "__m128i",
            },
            VecType {
                dtype: Dtype::Float32,
                value_cnt: 8,
                name: "vf8",
                native_type_name: "__m256",
            },
            VecType {
                dtype: Dtype::Float32,
                value_cnt: 4,
                name: "vf4",
                native_type_name: "__m128",
            },
            VecType {
                dtype: Dtype::Sint32,
                value_cnt: 8,
                name: "vsi8",
                native_type_name: "__m256i",
            },
            VecType {
                dtype: Dtype::Sint32,
                value_cnt: 4,
                name: "vsi4",
                native_type_name: "__m128i",
            },
            VecType {
                dtype: Dtype::Uint32,
                value_cnt: 8,
                name: "vui8",
                native_type_name: "__m256i",
            },
            VecType {
                dtype: Dtype::Uint32,
                value_cnt: 4,
                name: "vui4",
                native_type_name: "__m128i",
            },
            VecType {
                dtype: Dtype::Sint16,
                value_cnt: 16,
                name: "vsi16",
                native_type_name: "__m256i",
            },
            VecType {
                dtype: Dtype::Sint16,
                value_cnt: 8,
                name: "vsi8",
                native_type_name: "__m128i",
            },
            VecType {
                dtype: Dtype::Uint16,
                value_cnt: 16,
                name: "vui16",
                native_type_name: "__m256i",
            },
            VecType {
                dtype: Dtype::Uint16,
                value_cnt: 8,
                name: "vui8",
                native_type_name: "__m128i",
            },
            VecType {
                dtype: Dtype::Sint8,
                value_cnt: 32,
                name: "vsb32",
                native_type_name: "__m256i",
            },
            VecType {
                dtype: Dtype::Sint8,
                value_cnt: 16,
                name: "vsb16",
                native_type_name: "__m128i",
            },
            VecType {
                dtype: Dtype::Uint8,
                value_cnt: 32,
                name: "vub32",
                native_type_name: "__m256i",
            },
            VecType {
                dtype: Dtype::Uint8,
                value_cnt: 16,
                name: "vub16",
                native_type_name: "__m128i",
            },
            $( $extra, )*
        ];
    };
}

pub(crate) use define_x86_vec_types;
