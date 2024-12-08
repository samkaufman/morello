float horizontal_max_f32(__m256 input)
{
    __m128 v = _mm256_extractf128_ps(input, 1);
    v = _mm_max_ps(_mm256_castps256_ps128(input), v);
    v = __builtin_shufflevector(v, v, 2, 3, 0, 1);
    v = _mm_max_ps(v, __builtin_shufflevector(v, v, 1, 0, 3, 2));
    return _mm_cvtss_f32(v);
}
