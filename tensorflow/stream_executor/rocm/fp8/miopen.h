typedef enum {
  ...
  ...
  miopenFP8, /*!< 8-bit floating point 1:4:3 */
  miopenBF8, /*!< 8-bit binary floating point 1:5:2 */
  miopenImplicitType, /* !< Derive datatype implicitly from context */
} miopenDataType_t;


// The following MIOpen APIs should be enhanced to support the two new types above
//
// miopenSetTensorDescriptor
//   add support for dataType == [miopenFP8, miopenBF8]

// Enhance all routines below to handle tensors of type [miopenFP8, miopenBF8]
//
//  miopenConvolutionForwardGetWorkSpaceSize
//  miopenFindConvolutionForwardAlgorithm
//  miopenConvolutionForward
//
//  miopenConvolutionBackwardDataGetWorkSpaceSize
//  miopenFindConvolutionBackwardDataAlgorithm
//  miopenConvolutionBackwardData
//
//  miopenConvolutionBackwardWeightsGetWorkSpaceSize
//  miopenFindConvolutionBackwardWeightsAlgorithm
//  miopenConvolutionBackwardWeights
//
//
//  Support for following required if we intend to use immediate-mode  
//
//   miopenConvolutionForwardGetSolutionCount
//   miopenConvolutionForwardGetSolution
//   miopenConvolutionForwardCompileSolution
//   miopenConvolutionForwardImmediate
//
//   miopenConvolutionBackwardDataGetSolutionCount
//   miopenConvolutionBackwardDataGetSolution
//   miopenConvolutionBackwardDataCompileSolution
//   miopenConvolutionBackwardDataImmediate
//
//   miopenConvolutionBackwardWeightsGetSolutionCount
//   miopenConvolutionBackwardWeightsGetSolution
//   miopenConvolutionBackwardWeightsCompileSolution
//   miopenConvolutionBackwardWeightsImmediate
//
//
//  Support for following required if we intend to use Fusion API  
//
//   miopenCreateFusionPlan
//   miopenCreateOp* 
//   miopenCompileFusionPlan
//   miopenSetOpArgs* 
//   miopenExecuteFusionPlan


// Currently the miopen APIs do not provide a mechanism to explicitly specify the "compute_type"
// for a given operation. The "compute_type" is assumed to be the same as the type of the input
// and output "tensors" involved in the operation, which in turn are required to be all of the
// same type.
//
// This does not allow the framework to do something like pass float16 tensors to miopen
// convolution operation API, and at the same time instruct it to carry out the operation in
// float8 mode.  In order to carry out the operation in float8 mode, the input/output tensors
// must also have the float8 type!
//
// To facilitate the explicit specification of the "compute_type" for operations, the proposal
// is for a new "miopenDataType_t compute_type" field to be added to each operation descriptor
// (for example miopenConvolutionDescriptor). The default value for this field should be
// "miopenImplicitType", to preserve current behaviour / backward compatibility.
//
// For the initial satge of this project, we will only need this extra field for the
// miopenConvolutionDescriptor, and the only values supported for that field should be one of
// [miopenFP8, miopenBF8, miopenImplicitType]
//
// The intent here is to have MIOpen take float32/float16 tensors as input, and if the
// compute_type is a float8 type, then do float32/float16 --> float8 conversion on the input
// tensors, then do the convolution and finally do the float8 --> float32/float16 conversion
// on the output tensor.


// As we already mentioned, for all of the miopen supported operations, the datatype is required
// to be the same for all of the tensors involved in the operation.
//
// For float8 support, we need to be able to handle the scenario where the input tensor type(s) is
// different from the output tenor types(s) for the convolution operations. In this scenario,
// either the input type(s) or the output type(s) will match the "compute type". Basically this
// to support scenarios where the convolution op is on float32/float16 <---> float8 partition
// boundary, and the framework want to leverage MIOpen to do the required conversion.
