
// The proposal here is that framework code will use the existing "rocblas_gemm_ex" API.
// https://github.com/ROCmSoftwarePlatform/rocBLAS-internal/blob/develop/library/include/internal/rocblas-functions.h#L14622-L14657
// The "rocblas_gemm_ex" API should be extended to support the following scenarios

// Scenario 1: inputs are fp8, output is fp8, compute is fp8
// This will be communicated by
// rocblas_datatype_f8_r = a_type = b_type = c_type = d_type = compute_type


// Scenario 2A: inputs are fp16, output is fp16, compute is fp8
// This will be communicated by
// rocblas_datatype_f16_r = a_type = b_type = c_type = d_type
// rocblas_datatype_f8_r  = compute_type


// Scenario 2B: same as 2A, but with fp32 type instead of fp16
// Scenario 2C: same as 2A, but with bfloat16 type instead of fp16


// Scenario 3A: inputs are fp16, output is fp8, compute is fp8
// This will be communicated by
// rocblas_datatype_f16_r = a_type = b_type
// rocblas_datatype_f8_r  = c_type = d_type
// rocblas_datatype_f8_r  = compute_type


// Scenario 3B: same as 3A, but with fp32 type instead of fp16
// Scenario 3C: same as 3A, but with bfloat16 type instead of fp16


// Scenario 4A: inputs are fp8, output is fp16, compute is fp8
// This will be communicated by
// rocblas_datatype_f8_r  = a_type = b_type
// rocblas_datatype_f16_r = c_type = d_type
// rocblas_datatype_f8_r  = compute_type


// Scenario 4B: same as 4A, but with fp32 type instead of fp16
// Scenario 4C: same as 4A, but with bfloat16 type instead of fp16


// The above scenarios are in the order of priority, 1 is higher priority than 2 and so on
// Within a given scenario, A (fp16) is higher priority than B (fp32), and then C (bfloat16)


// QUESTION : Typically alpha and beta have the same type as compute_type.
//            Will that work here, given the low precision for fp8 type?
//            Or do we need to (implicitly?) hard-code the type for alpha / beta
//            as float (for the scenarios listed above)?


// Repeat all above scenarios for bf8 type
