//this file was generated by ../../../../../../scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__argmax__12.h"
#include "tracing.h"
#include "utils.h"

operator_status
execute_operator__ai_onnx__argmax__12(
    Onnx__NodeProto *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    //Onnx__TensorProto *i_data = searchInputByIndex(ctx, 0);

    // TRACE_TENSOR(2, true, i_data);

    // context_operator__ai_onnx__argmax__12 *op_ctx = ctx->executer_context;

    // int64_t axis = op_ctx->axis;
    // int64_t keepdims = op_ctx->keepdims;
    // int64_t select_last_index = op_ctx->select_last_index;

    // TRACE_VAR(2, true, axis, "%" PRId64);
    // TRACE_VAR(2, true, keepdims, "%" PRId64);
    // TRACE_VAR(2, true, select_last_index, "%" PRId64);

    //Onnx__TensorProto *o_reduced = searchOutputByIndex(ctx, 0);

    // TRACE_TENSOR(2, true, o_reduced);

    /* DO CALCULATION HERE */


    TRACE_EXIT(1);

    /* CHANGE RETURN CODE IF THIS EXECUTER IS VALID */
    return OP_ENOSYS;
    // return OP_OK;
}