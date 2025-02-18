//this file was generated by ../../../../../../scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__ISINF__10_H
# define OPERATOR_OPERATOR__AI_ONNX__ISINF__10_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'IsInf' version 10
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Map infinity to true and other values to false.
 * 
 * Constraint T1:
 *   Constrain input types to float tensors.
 *   Allowed Types: tensor_double, tensor_float
 * 
 * Constraint T2:
 *   Constrain output types to boolean tensors.
 *   Allowed Types: tensor_bool
 * Input T1 X:
 *   input
 *   Allowed Types: tensor_double, tensor_float
 * Output T2 Y:
 *   output
 *   Allowed Types: tensor_bool
 * Attribute INT detect_negative (optional):
 *   (Optional) Whether map negative infinity to true. Default to 1 so that
 *   negative infinity induces true. Set this attribute to 0 if negative
 *   infinity should be mapped to false.
 * 
 * Attribute INT detect_positive (optional):
 *   (Optional) Whether map positive infinity to true. Default to 1 so that
 *   positive infinity induces true. Set this attribute to 0 if positive
 *   infinity should be mapped to false.
*
* @since version 10
*
 * @see io/onnx/onnx/defs/tensor/defs.cc:2102
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#IsInf
*/

operator_status
prepare_operator__ai_onnx__isinf__10(
    Onnx__NodeProto *ctx
);

extern operator_info info_operator__ai_onnx__isinf__10;

typedef struct {
    int64_t detect_negative;
    int64_t detect_positive;

} context_operator__ai_onnx__isinf__10;

operator_status
execute_operator__ai_onnx__isinf__10(
    Onnx__NodeProto *ctx
);

# endif