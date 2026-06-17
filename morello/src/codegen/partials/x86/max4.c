float horizontal_max4_f32(__m128 input)
{
    const __m128 hiDual = _mm_movehl_ps(input, input);
    const __m128 maxDual = _mm_max_ps(input, hiDual);
    const __m128 hi = _mm_shuffle_ps(maxDual, maxDual, 0x1);
    const __m128 max = _mm_max_ss(maxDual, hi);
    return _mm_cvtss_f32(max);
}
