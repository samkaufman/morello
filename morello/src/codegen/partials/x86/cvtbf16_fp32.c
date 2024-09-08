inline __m256 cvtbf16_fp32(const __m128i a) {
  return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(a), 16));
}

inline void cvtbf16_fp32_256(const __m256i a, __m256 *o1, __m256 *o2) {
  *o1 = cvtbf16_fp32(_mm256_extractf128_si256(a, 1));
  *o2 = cvtbf16_fp32(_mm256_extractf128_si256(a, 0));
}
