#ifndef P2PUTILS_H
#define P2PUTILS_H

/* This file must be compiled with __SSE2__ defined. */

/* Output created .feat files in binary to speed up later steps. */
/* Improves CF performance by 5% */
#define BINARY_FEATURES              (1)

/* Potentially faster, but will reorder generated keypoints and make debugging more difficult. */
#define PARALLEL_KEYPOINT_GENERATION (1)

#define FAST_SIFT_DETECT             (0) /* Default behavior.  No error loss. */
#define FAST_SIFT_GRADIENT_UPDATE    (0) /* Faster, adds insignificant error. */
#define FAST_SIFT_CALC_KEYPOINTS     (0) /* Faster with ~1% error on 1% of the output */
#define FAST_SIFT_CALC_KEYPOINT_ORIENTATIONS (0)

/* EXP variants are no longer supported due to inaccuracy. */
/* Replaced by BetterFastExpSse below. */
#define    NO_FAST_EXP               (0)
#define    USE_FAST_EXP              (1)
#define    USE_FASTER_EXP            (2)
#define FAST_EXP_VARIANT             (USE_FAST_EXP)

#define    NO_FAST_ATAN2             (0)
#define    USE_FAST_ATAN2            (1)
#define FAST_ATAN2_VARIANT           (NO_FAST_ATAN2)

#ifndef WIN32
#undef USE_FP_FAST
#define USE_FP_FAST                  (0)
#endif

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>

#ifndef __SSE2__
#define __SSE2__
#endif

//#define __SSE4_1__
//#define __AVX2__

#define ALIGN_SIZE 16
#define GROUP_SIZE (4)

#include <emmintrin.h>
#define _Data __m128
#define _DataI __m128i
#define _CastIF _mm_castps_si128
#define _CastFI _mm_castsi128_ps
#define _ConvertFI _mm_cvtepi32_ps
#define _ConvertIF _mm_cvtps_epi32
#define _TruncateIF _mm_cvttps_epi32
#define _CmpGE(a,b) _mm_cmpge_ps(a, b)
#define _CmpGT(a,b) _mm_cmpgt_ps(a, b)
#define _CmpLT(a,b) _mm_cmplt_ps(a, b)
#define _CmpEQI(a,b) _mm_cmpeq_epi32(a, b)
#define _CmpLTI(a,b) _mm_cmplt_epi32(a, b)
#define _CmpGTI(a,b) _mm_cmpgt_epi32(a, b)
#define _Set _mm_set1_ps
#define _SetZero _mm_setzero_ps
#define _SetN(a,b,c,d) _mm_set_ps((d),(c),(b),(a))
#define _SetNDeltas(a,b) _SetN((a), ((a)+(b)), (a)+(b)*2.f, (a)+(b)*3.f)
#define _SetNMemory(a,b) _SetN((a)[0], (a)[b], (a)[2*(b)], (a)[3*(b)])
#define _Evens(a,b) _mm_shuffle_ps((a),(b), _MM_SHUFFLE(2, 0, 2, 0))
#define _Odds(a,b) _mm_shuffle_ps((a),(b), _MM_SHUFFLE(3, 1, 3, 1))
#define _Splat(a,b) _mm_shuffle_ps((a),(a), _MM_SHUFFLE((b),(b),(b),(b)))
#define _UnpackLow(a,b) _mm_unpacklo_ps((a), (b))
#define _UnpackHigh(a,b) _mm_unpackhi_ps((a), (b))
#define _SetI _mm_set1_epi32
#define _SetS _mm_set1_epi16
#define _Load _mm_loadu_ps
#define _LoadA _mm_load_ps
#define _LoadI _mm_load_si128
#define _vFirst _mm_cvtss_f32
#define _Store _mm_storeu_ps
#define _StoreA _mm_store_ps
#define _And _mm_and_ps
#define _AndI _mm_and_si128
#define _AndNot _mm_andnot_ps
#define _Or _mm_or_ps
#define _OrI _mm_or_si128
#define _Xor _mm_xor_ps
#define _ShiftLI _mm_slli_epi32
#define _ShiftRI _mm_srai_epi32
#define _ShiftRU _mm_srli_epi32
#define _Add _mm_add_ps
#define _AddI _mm_add_epi32
#define _Sub _mm_sub_ps
#define _SubI _mm_sub_epi32
#define _Mul _mm_mul_ps
#define _MulI _mm_mul_epi32
#define _Div _mm_div_ps
#define _Sqrt _mm_sqrt_ps
#define _Min _mm_min_ps
#define _Max _mm_max_ps
#define _Blend Blend
#define _AsArray(name, i) (name.m128_f32[i])
#define _AsArrayI(name, i) (name.m128i_i32[i])
#define _AsArrayS(name, i) (name.m128i_i16[i])

#if 1 //def __SSE4_1__ JPB WIP BUG
#include <smmintrin.h>
#define _Floor _mm_floor_ps
#endif

#ifdef __AVX2__
#include <immintrin.h>
#undef ALIGN_SIZE
#undef GROUP_SIZE
#undef _Data
#undef _DataI
#undef _CastFI
#undef _CastIF
#undef _ConvertFI
#undef _ConvertIF
#undef _TruncateIF
#undef _CmpGE
#undef _CmpGT
#undef _CmpLT
#undef _CmpEQI
#undef _CmpLT
#undef _CmpGEI
#undef _Set
#undef _SetZero
#undef _SetN
#undef _SetNDeltas
#undef _SetNMemory
#undef _Evens
#undef _Odds
#undef _Splat
#undef _UnpackLow
#undef _UnpackHigh
#undef _SetI
#undef _SetS
#undef _Load
#undef _LoadA
#undef _LoadI
#undef _vFirst
#undef _Store
#undef _StoreA
#undef _And
#undef _AndI
#undef _AndNot
#undef _Or
#undef _OrI
#undef _Xor
#undef _ShiftLI
#undef _ShiftRI
#undef _ShiftRU
#undef _Add
#undef _AddI
#undef _Sub
#undef _SubI
#undef _Mul
#undef _MulI
#undef _Div
#undef _Sqrt
#undef _Min
#undef _Max
#undef _Blend
#undef _AsArray
#undef _AsArrayI
#undef _AsArrayS
#undef _Floor
#define ALIGN_SIZE 32
#define GROUP_SIZE (8)
#define _Data __m256
#define _DataI __m256i
#define _DataH __m128
#define _CastFI _mm256_castsi256_ps
#define _CastIF _mm256_castps_si256
#define _ConvertFI _mm256_cvtepi32_ps
#define _ConvertIF _mm256_cvtps_epi32
#define _TruncateIF _mm256_cvttps_epi32
#define _CmpGE(a,b) _mm256_cmp_ps(a, b, _CMP_GE_OS)
#define _CmpGT(a,b) _mm256_cmp_ps(a, b, _CMP_GT_OS)
#define _CmpLT(a,b) _mm256_cmp_ps(a, b, _CMP_LT_OS)
#define _CmpEQI(a,b) _mm256_cmpeq_epi32(a, b)
#define _CmpGTI(a,b) _mm256_cmpgt_epi32(a, b)
#define _CmpLTI(a,b) _mm256_cmplt_epi32(a, b)
#define _Set _mm256_set1_ps
#define _SetZero _mm256_setzero_ps
#define _SetN(a,b,c,d,e,f,g,h) _mm256_set_ps((h),(g),(f),(e),(d),(c),(b),(a))
#define _SetNDeltas(a,b) _SetN((a), ((a)+(b)), (a)+(b)*2.f, (a)+(b)*3.f, (a)+(b)*4.f, (a)+(b)*5.f, (a)+(b)*6.f, (a)+(b)*7.f)
#define _SetNMemory(a,b) _SetN((a)[0], (a)[b], (a)[2*(b)], (a)[3*(b)], (a)[4*(b)], (a)[5*(b)], (a)[6*(b)], (a)[7*(b)])
/* JPB WIP #define _Evens(a,b) _mm_shuffle_ps((a),(b), _MM_SHUFFLE(2, 0, 2, 0))        */
/* JPB WIP #define _Odds(a,b) _mm_shuffle_ps((a),(b), _MM_SHUFFLE(3, 1, 3, 1))         */
/* JPB WIP #define _Splat(a,b) __mm_shuffle_ps((a),(a), _MM_SHUFFLE((b),(b),(b),(b)))  */
/* JPB WIP #define _UnpackLow(a,b) _mm_unpacklo_ps((a), (b))                           */
/* JPB WIP #define _UnpackHigh(a,b) _mm_unpackhi_ps((a), (b))                          */
#define _SetI _mm256_set1_epi32
#define _SetS _mm256_set1_epi16
#define _Load _mm256_loadu_ps
#define _LoadA _mm256_load_ps
#define _LoadI _mm256_load_si256  
#define _vFirst(a) _mm_cvtss_f32(_mm256_castps256_ps128(a))
#define _Store _mm256_storeu_ps
#define _StoreA _mm256_store_ps
#define _And _mm256_and_ps
#define _AndI _mm256_and_si256
#define _AndNot _mm256_andnot_ps
#define _Or _mm256_or_ps
#define _OrI _mm256_or_si256
#define _Xor _mm256_xor_ps
/* _Shifts are AVX2 only */
#define _ShiftLI _mm256_slli_epi32
#define _ShiftRI _mm256_srai_epi32
#define _ShiftRU _mm256_srli_epi32
#define _Add _mm256_add_ps
#define _AddI _mm256_add_epi32
#define _Sub _mm256_sub_ps
#define _SubI _mm256_sub_epi32
#define _Mul _mm256_mul_ps
#define _MulI _mm256_mul_epi32
#define _Div _mm256_div_ps
#define _Sqrt _mm256_sqrt_ps
#define _Min _mm256_min_ps
#define _Max _mm256_max_ps
#define _Blend Blend256
#define _AsArray(name, i) (name.m256_f32[i])
#define _AsArrayI(name, i) (name.m256i_i32[i])
#define _AsArrayS(name, i) (name.m256i_i16[i])
#define _Floor _mm256_floor_ps

#include <immintrin.h>
#endif // __AVX2__

#define _AddI3(a,b,c) _AddI((a),_AddI((b),(c)))

static __forceinline  float FastExpS(float x)
{
  /* Marginally faster than the table-driven method. */
  /* https://stackoverflow.com/questions/10552280/fast-exp-calculation-possible-to-improve-accuracy-without-losing-too-much-perfo/10792321#10792321 */
  volatile union {
      float f;
      unsigned int i;
  } cvt;

  /* exp(x) = 2^i * 2^f; i = floor (log2(e) * x), 0 <= f <= 1 */
  float t              = x * 1.442695041f;
  float fi             = floorf(t); /* JPB WIP BUG */
  float f              = t - fi;
  int i                = (int)fi;
  cvt.f                = (0.3371894346f * f + 0.657636276f) * f + 1.00172476f; /* compute 2^f */
  cvt.i                += (i << 23);                                           /* scale by 2^i */

  return cvt.f;
}

static __forceinline float FastExp3S(float x)  /* cubic spline approximation */
{
  /* Not a great substitute when accuracy is important. */
  /* Even BetterFastExpSse below works better. */
  /* https://stackoverflow.com/questions/53872265/how-do-you-process-exp-with-sse2 */
  /* Comparable speed to fast_expn, better accuracy? */
  union { float f; int i; } reinterpreter;

  reinterpreter.i      = (int)(12102203.0f*x) + 127*(1 << 23);
  int m = (reinterpreter.i >> 7) & 0xFFFF;          /* copy mantissa */
                                                    /* empirical values for small maximum relative error(8.34e-5) */
  reinterpreter.i      +=
      ((((((((1277*m) >> 14) + 14825)*m) >> 14) - 79749)*m) >> 11) - 626;
  return reinterpreter.f;
}

static float FastClampS(float x, float low, float high)
{
  return _vFirst(_mm_min_ps(_mm_max_ps(_Set(low), _Set(x)), _Set(high)));
}

static _Data FastClamp(_Data vX, _Data vLow, _Data vHigh)
{
  return _mm_min_ps(_mm_max_ps(vLow, vX), vHigh);
}

_Data __forceinline BetterFastExpSse(_Data vX)
{
  /* https://stackoverflow.com/questions/47025373/fastest-implementation-of-the-natural-exponential-function-using-sse */
  /* Of the several at the above link, this is the only consistently reliable version. */
  /* Clamping added manually. */
  static const _Data vLow
    = {-87.33654f, -87.33654f, -87.33654f, -87.33654f};
  static const _Data vHigh
    = {88.72283f, 88.72283f, 88.72283f, 88.72283f};

  vX                    = FastClamp(vX, vLow, vHigh);

  const _Data a         = _mm_set1_ps ((1 << 22) / float(M_LN2)); /* to get exp(x / 2) */
  const _DataI b        = _mm_set1_epi32 (127 * (1 << 23));       /* NB: zero shift! */
  _DataI r              = _mm_cvtps_epi32 (_mm_mul_ps (a, vX));
  _DataI s              = _mm_add_epi32 (b, r);
  _DataI t              = _mm_sub_epi32 (b, r);

  return _mm_div_ps (_mm_castsi128_ps (s), _mm_castsi128_ps (t));
}

static inline _DataI muly(const _DataI &a, const _DataI &b)
{
  _DataI tmp1           = _mm_mul_epu32(a,b); /* mul 2,0*/
  _DataI tmp2           = _mm_mul_epu32( _mm_srli_si128(a,4), _mm_srli_si128(b,4)); /* mul 3,1 */

  return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, _MM_SHUFFLE (0,0,2,0)), _mm_shuffle_epi32(tmp2, _MM_SHUFFLE (0,0,2,0))); /* shuffle results to [63..0] and pack */
}

static inline _Data recip_float4_single(_Data x) {
  _Data res             = _mm_rcp_ps(x);
  _Data muls            = _mm_mul_ps(x, _mm_mul_ps(res, res));

  return res            =  _mm_sub_ps(_mm_add_ps(res, res), muls);
}

static __forceinline _Data FasterExp(_Data vX)
{
  return BetterFastExpSse(vX);
}

static  __forceinline _Data FastExp(_Data vX)
{
  return BetterFastExpSse(vX);
}

static  __forceinline _Data FastExpAlwaysNegative(_Data vX)
{
  /* https://stackoverflow.com/questions/47025373/fastest-implementation-of-the-natural-exponential-function-using-sse */
  /* Of the several at the above link, this is the only consistently reliable version. */
  /* Clamping added manually. */
  static const _Data vLow
    = {-87.33654f, -87.33654f, -87.33654f, -87.33654f};

  vX                    = _Max(vX, vLow);

  const __m128 a        = _mm_set1_ps ((1 << 22) / float(M_LN2)); /* to get exp(x / 2) */
  const __m128i b       = _mm_set1_epi32 (127 * (1 << 23));       /* NB: zero shift! */
  __m128i r             = _mm_cvtps_epi32 (_mm_mul_ps (a, vX));
  __m128i s             = _mm_add_epi32 (b, r);
  __m128i t             = _mm_sub_epi32 (b, r);

  return _mm_div_ps (_mm_castsi128_ps (s), _mm_castsi128_ps (t));
}

static  __forceinline _Data FastExpAlwaysPositive(_Data vX)
{
  /* https://stackoverflow.com/questions/47025373/fastest-implementation-of-the-natural-exponential-function-using-sse */
  /* Of the several at the above link, this is the only consistently reliable version. */
  /* Clamping added manually. */
  static const _Data vHigh
    = {88.72283f, 88.72283f, 88.72283f, 88.72283f};

  vX                    = _Min(vX, vHigh);

  const __m128 a        = _mm_set1_ps ((1 << 22) / float(M_LN2)); /* to get exp(x / 2) */
  const __m128i b       = _mm_set1_epi32 (127 * (1 << 23));       /* NB: zero shift! */
  __m128i r             = _mm_cvtps_epi32 (_mm_mul_ps (a, vX));
  __m128i s             = _mm_add_epi32 (b, r);
  __m128i t             = _mm_sub_epi32 (b, r);

  return _mm_div_ps (_mm_castsi128_ps (s), _mm_castsi128_ps (t));
}

#ifdef __SSE4_1__
static __forceinline _Data Blend(_Data a, _Data b, _Data mask)
{
  return _mm_blendv_ps(a, b, mask);
}
#else
static __forceinline _Data Blend(_Data a, _Data b, _Data mask)
{
  _Data const andMask   = _And(mask, b);            /* and_mask = ( mask & b ) */
  _Data const andNotMask
                        = _AndNot(mask, a);         /* andnot_mask = (~mask & a) */

  return _Or(andMask, andNotMask);                  /* result = ( mask & b ) | ( ~mask & a ) */
}
#endif

#ifdef __AVX2__
static __forceinline _Data Blend256( _Data a, _Data b, _Data mask )
{
  return _mm256_blendv_ps( a, b, mask );
}
#endif

static __forceinline _Data FastAtan(_Data x)
{
  /* https://blasingame.engr.tamu.edu/z_zCourse_Archive/P620_18C/P620_zReference/PDF_Txt_Hst_Apr_Cmp_(1955).pdf */
  _Data const a1        = _Set(0.99997726f);
  _Data const a3        = _Set(-0.33262347f);
  _Data const a5        = _Set(0.19354346f);
  _Data const a7        = _Set(-0.11643287f);
  _Data const a9        = _Set(0.05265332f);
  _Data const a11       = _Set(-0.01172120f);

  _Data const x_sq = _Mul(x, x);
  _Data result = a11;
  /* JPB WIP fmadd */
  result                = _Add(_Mul(x_sq, result), a9);
  result                = _Add(_Mul(x_sq, result), a7);
  result                = _Add(_Mul(x_sq, result), a5);
  result                = _Add(_Mul(x_sq, result), a3);
  result                = _Add(_Mul(x_sq, result), a1);

  return _Mul(x, result);
}

static __forceinline _Data FastATan2(_Data y, _Data x)
{
  /* Not bitwise compatible with vl_fast_atan2_*/
  _Data const vPi      = _Set((float) M_PI);
  _Data const vPi2     = _Set((float) M_PI_2);
  _Data const vAbsMask = _CastFI(_SetI(0x7FFFFFFF));
  _Data const vSignMask
                       = _CastFI(_SetI(0x80000000));
  _Data const vSwapMask
                       = _CmpGT(
    _And( y, vAbsMask ),  /* |y| */
    _And( x, vAbsMask )   /* |x| */
  );
  /* pick the lowest between |y| and |x| for each number */
  _Data const vLow     = _Blend(y, x, vSwapMask);
  _Data const vHigh    = _Blend(x, y, vSwapMask);
  _Data const vAtan    = _Div(
    vLow,
    vHigh
  );

  _Data vResult = FastAtan(vAtan);

  vResult = _Blend(
    vResult,
    _Sub(
      _Or(vPi2, _And(vAtan, vSignMask)),
      vResult
    ),
    vSwapMask
  );

  _Data const vXSignMask = _CastFI(_ShiftRI(_CastIF(x), 31));

  return _Add(
    _And(
      _Xor(vPi, _And(vSignMask, y)),
       vXSignMask
    ),
    vResult
  );
}

#ifndef VL_PI
#define VL_PI 3.141592653589793
#endif

static __forceinline float FastSqrtS(float x)
{
  /* Bare sqrt, avoid setting global state. */
  return _vFirst(_mm_sqrt_ss(_Set(x)));
}

static __forceinline float FastRSqrtS(float x) {
  //taken from http://stackoverflow.com/q/14752399/556899
  float res            = _vFirst(_mm_rsqrt_ss(_Set(x))); 

  return (0.5f*res) * (3.f-((x*res)*res));
}

/* Remember this is a C-file (no templates) */
static __forceinline float FastMinF(float a, float b)
{
  return a < b ? a : b;
}

static __forceinline _Data Mod2PILimited(_Data x)
{
  /* Perform a limited mod on the components of x.
   * x is in the range [-4PI, +4PI] */
  static _Data const vTwoPI   = _Set((float)(2. * VL_PI));

  _Data needsReduction = _CmpGE(x, vTwoPI);
  _Data vOffset        = _And(needsReduction, vTwoPI);  /* 0.0 or 2*Pi */
  _Data const vResult  = _Sub(x, vOffset);

  needsReduction       = _CmpLT(x, _SetZero());
  vOffset              = _And(needsReduction, vTwoPI);  /* 0.0 or 2*Pi */

  return _Add(vResult, vOffset);
}

static __forceinline float FastAbsS(float x)
{
  return std::max(0.f - x, x);
}

static __forceinline _Data FastAbs(_Data vX)
{
  /* https://stackoverflow.com/questions/5508628/how-to-absolute-2-double-or-4-floats-using-sse-instruction-set-up-to-sse4 */
  static const _Data sign_mask = _Set(-0.f); // -0.f = 1 << 31

  return _AndNot(sign_mask, vX);
}

static __forceinline _Data FastACos(_Data vX)
{
  static const _Data vOne = {1.f, 1.f, 1.f, 1.f};

  _Data vNegate = Blend(vX, vOne, _CmpLT(vX, _mm_setzero_ps()));

  vX = FastAbs(vX);

  static const _Data vC[] = {
    {-0.0187293f, -0.0187293f, -0.0187293f, -0.0187293f},
    {0.0742610f, 0.0742610f, 0.0742610f, 0.0742610f},
    {0.2121144f, 0.2121144f, 0.2121144f, 0.2121144f},
    {1.5707288f, 1.5707288f, 1.5707288f, 1.5707288f}
  };

  static const _Data vPI = {3.14159265358979f, 3.14159265358979f, 3.14159265358979f, 3.14159265358979f};

  _Data vResult = vC[0];

  vResult = _Mul(vResult, vX);

  vResult = _Add(vResult, vC[1]);
  vResult = _Mul(vResult, vX);

  vResult = _Add(vResult, vC[2]);
  vResult = _Mul(vResult, vX);

  vResult = _Add(vResult, vC[3]);
  vResult = _Mul(vResult, _Sqrt(_Sub(vOne, vX)));
  vResult = _Sub(vResult, _Mul(_Add(vNegate, vNegate), vResult));

  return _Add(_Mul(vNegate, vPI), vResult);
}

static __forceinline float FastACosS(float x) {
  /* https://developer.download.nvidia.com/cg/acos.html */
  /* Handbook of Mathematical Functions */
  /* M. Abramowitz and I.A. Stegun, Ed. */
  /* Absolute error <= 6.7e-5           */

  float negate = float(x < 0.f);
  x = FastAbsS(x);
  float ret = -0.0187293f;
  ret = ret * x;

  ret = ret + 0.0742610f;
  ret = ret * x;

  ret = ret - 0.2121144f;
  ret = ret * x;

  ret = ret + 1.5707288f;
  ret = ret * FastSqrtS(1.0f-x);
  ret = ret - 2.f * negate * ret;
  return negate * 3.14159265358979f + ret;
}

static __forceinline _Data GradCalc2(_Data y, _Data x)
{
  /* An SSE2-comparable variant of vl_mod_2pi_f(vl_fast_atan2_f (gy, gx) + 2*VL_PI) */
  _Data const vTwoPI   = _Set((float)(2. * VL_PI));

  _Data const vC3      = _Set(0.1821f);
  _Data const vC1      = _Set(0.9675f);
  _Data const vAbsY    = FastAbs(_Add(y, _Set(1.19209290E-07F)));
  
  _Data const vHighBit = _CastFI(_SetI(0x80000000));

  _Data const vNum     = _Sub(x, _Or(vAbsY, _And(x, vHighBit)));
  _Data const vDen     = _Add(vAbsY, _Xor(x, _And(x, vHighBit)));
  _Data vAngle         = Blend(_Set((float)(3.*VL_PI/4.)), _Set((float)(VL_PI/4.)), _CmpGE(x, _SetZero()));
  _Data const vR       = _Div(vNum, vDen);
  vAngle               = _Add(
                            vAngle,
                            _Mul(_Sub(_Mul(vC3, _Mul(vR, vR)), vC1), vR)
                         );

  _Data const atan2    = _Xor(vAngle, _And(y, vHighBit));

  return Mod2PILimited(_Add(atan2, vTwoPI));
}

static __forceinline float Mod2PILimitedS(float x)
{
  _Data const vTmp = Mod2PILimited(_Set(x));

  return _vFirst( vTmp );
}

static __forceinline _Data Floor(_Data vX)
{
  /* https://gist.github.com/mmozeiko/56db3df14ab380152d6875383d0f4afd */
#ifdef __SSE4_1__
  return _Floor(x);
#else
  _DataI const v0       = _SetI(0);
  _DataI const v1       = _CmpEQI(v0, v0);
  _DataI const ji       = _ShiftRU(v1, 25);
  _DataI const tmp      = _ShiftLI(ji, 23); /* I edited this (Added tmp) not sure about it */
  _Data j               = _CastFI(tmp);     /* create vector 1.0f - I edited this not sure about it */
  _DataI const i        = _TruncateIF(vX);
  _Data const fi        = _ConvertFI(i);
  _Data const igx       = _CmpGT(fi, vX);
  j                     = _And(igx, j);

  return _Sub(fi, j);
#endif
}

static __forceinline int AllZerosI(_DataI x)
{
#ifdef __SSE4_1__
  return _mm_testz_si128(x, x);
#else
  return _mm_movemask_epi8(_mm_cmpeq_epi8(x, _mm_setzero_si128())) == 0xFFFF;
#endif
}

static __forceinline _DataI FastMod8(_DataI x)
{
  /* This a -signed- mod 8
   * x % 2n == x < 0 ? x | ~(2n - 1) : x & (2n - 1)
   * x % 8 ==  x < 0 ? x | ~(4-1) : x & 3 */

  _DataI xmm2           = _SetI(7);
  _DataI vZero          = _SetI(0);
  _DataI xmm1           = _CmpGTI(vZero, x);
  _DataI xmm0           = x;
  xmm1                  = _AndI(xmm1, xmm2);
  xmm0                  = _AddI(xmm0, xmm1);
  xmm0                  = _AndI(xmm0, xmm2);

  return _SubI( xmm0, xmm1 );
  /*
  https://godbolt.org/z/d99T564qo
  movdqu  xmm3, XMMWORD PTR [rdi]
  pxor    xmm1, xmm1
  movdqa  xmm2, XMMWORD PTR .LC0[rip]
  pcmpgtd xmm1, xmm3
  movdqa  xmm0, xmm3
  pand    xmm1, xmm2
  paddd   xmm0, xmm1
  pand    xmm0, xmm2
  psubd   xmm0, xmm1
  movups  XMMWORD PTR [rdi], xmm0
  */
}

// https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-sse-vector-sum-or-other-reduction
#ifdef __AVX__
static __forceinline float hsum_ps_sse3(_Data v)
{
  _DataH shuf           = _mm_movehdup_ps(v);        /* broadcast elements 3,1 to 2,0 */
  _DataH sums           = _Add(v, shuf);
  shuf                  = _mm_movehl_ps(shuf, sums); /* high half -> low half */
  sums                  = _mm_add_ss(sums, shuf);
  return _vFirst(sums);
}

static __forceinline float FastHsum(_Data v)
{
  _DataH vlow  = _mm256_castps256_ps128(v);
  _DataH vhigh = _mm256_extractf128_ps(v, 1); // high 128
  _DataH v128  = _mm_add_ps(vlow, vhigh);     // add the low 128

  return hsum_ps_sse3(v128);             // and inline the sse3 version, which is optimal for AVX
  // (no wasted instructions, and all of them are the 4B minimum)
}

static __forceinline float hprod_ps_sse3(_Data v)
{
  _DataH shuf           = _mm_movehdup_ps(v);        /* broadcast elements 3,1 to 2,0 */
  _DataH products       = _Mul(v, shuf);
  shuf                  = _mm_movehl_ps(shuf, products); /* high half -> low half */
  products              = _mm_mul_ss(products, shuf);

  return _vFirst(products);
}

static __forceinline float FastHProduct(_Data v)
{
  _DataH vlow  = _mm256_castps256_ps128(v);
  _DataH vhigh = _mm256_extractf128_ps(v, 1); // high 128
  _DataH v128  = _mm_mul_ps(vlow, vhigh);     // multiply the low 128

  return hprod_ps_sse3(v128);                 // and inline the sse3 version, which is optimal for AVX
  // (no wasted instructions, and all of them are the 4B minimum)
}

#elif defined(__SSE3__)
static __forceinline float FastHsum(_Data v)
{
  _Data shuf            = _mm_movehdup_ps(v);        /* broadcast elements 3,1 to 2,0 */
  _Data sums            = _Add(v, shuf);
  shuf                  = _mm_movehl_ps(shuf, sums); /* high half -> low half */
  sums                  = _mm_add_ss(sums, shuf);

  return _vFirst(sums);
}

static __forceinline float FastHProduct(_Data v)
{
  _Data shuf            = _mm_movehdup_ps(v);            /* broadcast elements 3,1 to 2,0 */
  _Data products        = _Mul(v, shuf);
  shuf                  = _mm_movehl_ps(shuf, products); /* high half -> low half */
  products              = _mm_add_ss(products, shuf);

  return _vFirst(sums);
}
#else
static __forceinline float FastHsum(_Data v)
{
                                                                         /* v = [ D C | B A ] */
  _Data shuf            = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1)); /* [ C D | B A ] */
  _Data sums            = _Add(v, shuf);                                 /* sums = [ D+C C+D | B+A A+B ] */
  shuf                  = _mm_movehl_ps(shuf, sums);                     /*  [   C   D | D+C C+D ] let the compiler avoid a mov by reusing shuf */
  sums                  = _mm_add_ss(sums, shuf);

  return _vFirst(sums);
}

static __forceinline float FastHProduct(_Data v)
{
  /* v = [ D C | B A ] */
  _Data shuf            = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1)); /* [ C D | B A ] */
  _Data products        = _Mul(v, shuf);                                 /* products = [ D*C C*D | B*A A*B ] */
  shuf                  = _mm_movehl_ps(shuf, products);                 /*  [   C   D | D*C C*D ] let the compiler avoid a mov by reusing shuf */
  products              = _mm_mul_ss(products, shuf);

  return _vFirst(products);
}
#endif

static __forceinline float FastDot4(_Data v1, _Data v2)
{
#if 0 //def __SSE4_1__
  /* JPB WIP OPT */
  return _vFirst(_mm_dp_ps(v1, v2, 0xFF));
#else
  __m128 r1 = _mm_mul_ps(v1, v2);
  __m128 shuf   = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 3, 0, 1));
  __m128 sums   = _mm_add_ps(r1, shuf);
  shuf          = _mm_movehl_ps(shuf, sums);
  sums          = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
#endif
}
#endif /* P2PUTILS_H */
