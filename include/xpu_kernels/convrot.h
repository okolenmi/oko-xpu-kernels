// xpu-int8: fused ConvRot INT8 primitives for Intel Arc (Xe2) via SYCL.
//
// Public C API. Both entry points take raw device pointers so they can be called
// directly on torch XPU tensors via ctypes / data_ptr().
//
//   xpuint8_convrot_quantize   bf16 [M,K] -> int8 [M,K] + fp32 [M] row scales
//                              (Kronecker-Hadamard rotation fused with row-wise
//                               INT8 quantization; replaces the eager backend's
//                               bmm + abs + amax + div + round + clamp + cast)
//
//   xpuint8_convrot_rotate_bf16  bf16 [M,K] -> bf16 [M,K]
//                              (rotation only, no quantization). This is what
//                              the BF16 path needs: ConvRot stores the weights
//                              pre-rotated, so dequantized BF16 weights are
//                              already rotated and only the activation has to
//                              be. Also the fair baseline when comparing INT8
//                              against BF16 -- a BF16 comparison that omits this
//                              rotation is not apples to apples.
//
//   xpuint8_dequant_epilogue   int32 [M,N] -> bf16 [M,N]
//                              (per-row x scale, per-column weight scale, bias)
#ifndef XPU_INT8_H
#define XPU_INT8_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Device selected on first use; override with XPUINT8_DEVICE_SELECTOR.
int xpuint8_init(void);

// Human-readable device name, or NULL if no device.
const char *xpuint8_device_name(void);

// Group size used by the fused rotation. Must be a power of 4 (ConvRot rule).
// The implementation is specialised for 256; other sizes fall back internally.
int xpuint8_convrot_groupsize(void);

// Rotate + quantize. x is bfloat16 [M,K]; q is int8 [M,K]; scale is fp32 [M].
// Returns 0 on success, non-zero on failure (message in xpuint8_last_error).
int xpuint8_convrot_quantize(const void *x, int8_t *q, float *scale, int64_t M,
                             int64_t K);

// Stream variants enqueue asynchronously on a caller-owned SYCL queue. They
// must be passed the queue pointer exposed by the owning framework stream.
int xpuint8_convrot_quantize_stream(const void *sycl_queue, const void *x, int8_t *q,
                                    float *scale, int64_t M, int64_t K);

// Apply a supported elementwise activation before rotation and quantization.
// activation: 1 = GELU with tanh approximation, x is bf16 [M,K];
//             2 = SwiGLU, x is bf16 [M,2*K] laid out as [gate, up].
int xpuint8_convrot_quantize_act(const void *x, int8_t *q, float *scale, int64_t M,
                                 int64_t K, int32_t activation);
int xpuint8_convrot_quantize_act_stream(const void *sycl_queue, const void *x, int8_t *q,
                                        float *scale, int64_t M, int64_t K,
                                        int32_t activation);

// Rotate only, bf16 -> bf16. x and xr are bfloat16 [M,K].
int xpuint8_convrot_rotate_bf16(const void *x, void *xr, int64_t M, int64_t K);
int xpuint8_convrot_rotate_bf16_stream(const void *sycl_queue, const void *x, void *xr,
                                      int64_t M, int64_t K);

// Dequantize + bias. acc is int32 [M,N]; out is bfloat16 [M,N].
// bias may be NULL. All pointers must be device-accessible.
int xpuint8_dequant_epilogue(const int32_t *acc, void *out, const float *x_scale,
                             const float *w_scale, const float *bias, int64_t M,
                             int64_t N);
int xpuint8_dequant_epilogue_stream(const void *sycl_queue, const int32_t *acc,
                                    void *out, const float *x_scale,
                                    const float *w_scale, const float *bias,
                                    int64_t M, int64_t N);
// Same, for a [M,N] column slice of a [M,acc_row_stride] int32 buffer: reads
// acc + m*acc_row_stride instead of acc + m*N, so a fused projection's
// accumulator can be dequantized one output group at a time without a copy.
// Bit-identical to running the contiguous variant on the slice.
int xpuint8_dequant_epilogue_slice(const int32_t *acc, void *out, const float *x_scale,
                                   const float *w_scale, const float *bias, int64_t M,
                                   int64_t N, int64_t acc_row_stride);
int xpuint8_dequant_epilogue_slice_stream(const void *sycl_queue, const int32_t *acc,
                                          void *out, const float *x_scale,
                                          const float *w_scale, const float *bias,
                                          int64_t M, int64_t N, int64_t acc_row_stride);
// Fused dequantize + activation + rotate + quantize. acc is int32 [M,2*K]: the
// first K columns are the activation, the second K the gate. Applies
// activation 3 (situ) to the dequantized pair, then rotates and quantizes the
// first half into q (int8 [M,K]) with per-row scale. Equivalent to
// xpuint8_dequant_epilogue followed by xpuint8_convrot_quantize_act(..., 3),
// without the bf16 round trip. x_scale is [M], w_scale is [2*K].
int xpuint8_dequant_act_quantize(const void *acc, int8_t *q, float *scale,
                                 const float *x_scale, const float *w_scale,
                                 int64_t M, int64_t K, int32_t activation);
int xpuint8_dequant_act_quantize_stream(const void *sycl_queue, const void *acc,
                                        int8_t *q, float *scale,
                                        const float *x_scale, const float *w_scale,
                                        int64_t M, int64_t K, int32_t activation);

const char *xpuint8_last_error(void);

#ifdef __cplusplus
}
#endif
#endif  // XPU_INT8_H
