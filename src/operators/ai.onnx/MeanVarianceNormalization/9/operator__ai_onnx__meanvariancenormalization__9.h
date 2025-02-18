//this file was generated by ../../../../../../scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__MEANVARIANCENORMALIZATION__9_H
# define OPERATOR_OPERATOR__AI_ONNX__MEANVARIANCENORMALIZATION__9_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'MeanVarianceNormalization' version 9
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * A MeanVarianceNormalization Function: Perform mean variance normalization
 *       on the input tensor X using formula: <br/> ``` (X-EX)/sqrt(E(X-EX)^2) ```
 * 
 * Constraint T:
 *   Constrain input and output types to all numeric tensors.
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * Input T X:
 *   Input tensor
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * Output T Y:
 *   Output tensor
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * Attribute INTS axes (optional):
 *   A list of integers, along which to reduce. The default is to caculate
 *   along axes [0,2,3] for calculating mean and variance along each channel.
 *   Two variables with the same C-coordinate are associated with the same mean
 *   and variance.
*
* @since version 9
*
 * @see io/onnx/onnx/defs/nn/defs.cc:2129
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#MeanVarianceNormalization
*/

operator_status
prepare_operator__ai_onnx__meanvariancenormalization__9(
    Onnx__NodeProto *ctx
);

extern operator_info info_operator__ai_onnx__meanvariancenormalization__9;

typedef struct {
    size_t n_axes;
    int64_t* axes;

} context_operator__ai_onnx__meanvariancenormalization__9;

operator_status
execute_operator__ai_onnx__meanvariancenormalization__9(
    Onnx__NodeProto *ctx
);

# endif