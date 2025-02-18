//this file was generated by ../../../../../../scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__AI_ONNX__TOPK__1_H
# define OPERATOR_OPERATOR__AI_ONNX__TOPK__1_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * ai.onnx operator 'TopK' version 1
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Retrieve the top-K elements along a specified axis. Given an input tensor of
 * shape [a_1, a_2, ..., a_n, r] and integer argument k, return two outputs:
 *   -Value tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
 *     which contains the values of the top k elements along the specified axis
 *   -Index tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] which
 *    contains the indices of the top k elements (original indices from the input
 *    tensor).
 * Given two equivalent values, this operator uses the indices along the axis  as
 *  a tiebreaker. That is, the element with the lower index will appear first.
 * 
 * Constraint T:
 *   Constrain input and output types to float tensors.
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * 
 * Constraint I:
 *   Constrain index tensor to int64
 *   Allowed Types: tensor_int64
 * Input T X:
 *   Tensor of shape [a_1, a_2, ..., a_n, r]
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * Output T Values:
 *   Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
 *   containing top K values from the input tensor
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * 
 * Output I Indices:
 *   Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
 *   containing the corresponding input tensor indices for the top K values.
 *   Allowed Types: tensor_int64
 * Attribute INT axis (optional):
 *   Dimension on which to do the sort.
 * 
 * Attribute INT k :
 *   Number of top elements to retrieve
*
* @since version 1
*
 * @see io/onnx/onnx/defs/math/old.cc:1497
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#TopK
*/

operator_status
prepare_operator__ai_onnx__topk__1(
    Onnx__NodeProto *ctx
);

extern operator_info info_operator__ai_onnx__topk__1;

typedef struct {
    int64_t axis;
    int64_t k;

} context_operator__ai_onnx__topk__1;

operator_status
execute_operator__ai_onnx__topk__1(
    Onnx__NodeProto *ctx
);

# endif