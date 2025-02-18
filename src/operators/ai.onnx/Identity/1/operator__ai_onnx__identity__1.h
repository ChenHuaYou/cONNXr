//this file was generated by ../../../../../../scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__IDENTITY__1_H
# define OPERATOR_OPERATOR__AI_ONNX__IDENTITY__1_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'Identity' version 1
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Identity operator
 * 
 * Constraint T:
 *   Constrain input and output types to all tensor types.
 *   Allowed Types: tensor_bool, tensor_complex128, tensor_complex64,
 *                  tensor_double, tensor_float, tensor_float16, tensor_int16,
 *                  tensor_int32, tensor_int64, tensor_int8, tensor_string,
 *                  tensor_uint16, tensor_uint32, tensor_uint64, tensor_uint8
 * Input T input:
 *   Input tensor
 *   Allowed Types: tensor_bool, tensor_complex128, tensor_complex64,
 *                  tensor_double, tensor_float, tensor_float16, tensor_int16,
 *                  tensor_int32, tensor_int64, tensor_int8, tensor_string,
 *                  tensor_uint16, tensor_uint32, tensor_uint64, tensor_uint8
 * Output T output:
 *   Tensor to copy input into.
 *   Allowed Types: tensor_bool, tensor_complex128, tensor_complex64,
 *                  tensor_double, tensor_float, tensor_float16, tensor_int16,
 *                  tensor_int32, tensor_int64, tensor_int8, tensor_string,
 *                  tensor_uint16, tensor_uint32, tensor_uint64, tensor_uint8

*
* @since version 1
*
 * @see io/onnx/onnx/defs/tensor/defs.cc:1825
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#Identity
*/

operator_status
prepare_operator__ai_onnx__identity__1(
    Onnx__NodeProto *ctx
);

extern operator_info info_operator__ai_onnx__identity__1;

typedef struct {
// no attributes
} context_operator__ai_onnx__identity__1;

operator_status
execute_operator__ai_onnx__identity__1(
    Onnx__NodeProto *ctx
);

# endif