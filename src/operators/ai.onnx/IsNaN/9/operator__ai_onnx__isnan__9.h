//this file was generated by ../../../../../../scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__ISNAN__9_H
# define OPERATOR_OPERATOR__AI_ONNX__ISNAN__9_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'IsNaN' version 9
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Returns which elements of the input are NaN.
 * 
 * Constraint T1:
 *   Constrain input types to float tensors.
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * 
 * Constraint T2:
 *   Constrain output types to boolean tensors.
 *   Allowed Types: tensor_bool
 * Input T1 X:
 *   input
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * Output T2 Y:
 *   output
 *   Allowed Types: tensor_bool

*
* @since version 9
*
 * @see io/onnx/onnx/defs/tensor/defs.cc:2066
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#IsNaN
*/

operator_status
prepare_operator__ai_onnx__isnan__9(
    Onnx__NodeProto *ctx
);

extern operator_info info_operator__ai_onnx__isnan__9;

typedef struct {
// no attributes
} context_operator__ai_onnx__isnan__9;

operator_status
execute_operator__ai_onnx__isnan__9(
    Onnx__NodeProto *ctx
);

# endif