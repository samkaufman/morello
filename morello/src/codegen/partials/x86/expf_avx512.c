// Adapted from the AVX2 implementation by SO user "njuffa":
// https://stackoverflow.com/a/49090523/110389
inline __m512 exp512_ps(__m512 x) {
  __m512 t, f, p, r;

  const __m512 l2e = _mm512_set1_ps(1.442695041f);    /* log2(e) */
  const __m512 l2h = _mm512_set1_ps(-6.93145752e-1f); /* -log(2)_hi */
  const __m512 l2l = _mm512_set1_ps(-1.42860677e-6f); /* -log(2)_lo */
  /* coefficients for core approximation to exp() in [-log(2)/2, log(2)/2] */
  const __m512 c0 = _mm512_set1_ps(0.041944388f);
  const __m512 c1 = _mm512_set1_ps(0.168006673f);
  const __m512 c2 = _mm512_set1_ps(0.499999940f);
  const __m512 c3 = _mm512_set1_ps(0.999956906f);
  const __m512 c4 = _mm512_set1_ps(0.999999642f);

  /* exp(x) = 2^r * e^f; r = rint(log2(e) * x), f = x - log(2) * r */
  t = _mm512_mul_ps(x, l2e); /* t = log2(e) * x */
  r = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

  f = _mm512_fmadd_ps(r, l2h, x); /* x - log(2)_hi * r */
  f = _mm512_fmadd_ps(r, l2l, f); /* f = x - log(2)_hi * r - log(2)_lo * r */

  /* p ~= exp(f), -log(2)/2 <= f <= log(2)/2 */
  p = c0;                        /* c0 */
  p = _mm512_fmadd_ps(p, f, c1); /* c0*f+c1 */
  p = _mm512_fmadd_ps(p, f, c2); /* (c0*f+c1)*f+c2 */
  p = _mm512_fmadd_ps(p, f, c3); /* ((c0*f+c1)*f+c2)*f+c3 */
  p = _mm512_fmadd_ps(p, f, c4); /* (((c0*f+c1)*f+c2)*f+c3)*f+c4 ~= exp(f) */

  /* exp(x) = 2^r * p */
  return _mm512_scalef_ps(p, r);
}
