
typedef struct rocblas_fp8 {
  uint8_t data;
} rocblas_fp8;

typedef struct rocblas_bf8 {
  uint8_t data;
} rocblas_bf8;


typedef enum rocblas_datatype_
{
 ...
 ...

 // Note use of "f8" instead of "fp8" is intentional. This is to
 // keep things consistent with the rest of the enum literals in this type
 
 rocblas_datatype_f8_r , /**< 8 bit floating point, real */
 rocblas_datatype_bf8_r , /**< 8 bit bfloat, real */

 // QUESTION : Do we need to create the complex equivalents?
 //            They exist for f16/bf16 , but do not know how/where they are used!
 
} rocblas_datatype;
