//this file was generated by ../../../../../../scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__concat__11.h"
#include "tracing.h"
#include "utils.h"

operator_status
execute_operator__ai_onnx__concat__11(
    Onnx__NodeProto *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    //Onnx__TensorProto *i_inputs = searchInputByIndex(ctx, 0);

    // TRACE_TENSOR(2, true, i_inputs);

    // context_operator__ai_onnx__concat__11 *op_ctx = ctx->executer_context;

    // int64_t axis = op_ctx->axis;

    // TRACE_VAR(2, true, axis, "%" PRId64);

    //Onnx__TensorProto *o_concat_result = searchOutputByIndex(ctx, 0);

    // TRACE_TENSOR(2, true, o_concat_result);

    /* DO CALCULATION HERE */


    TRACE_EXIT(1);

    /* CHANGE RETURN CODE IF THIS EXECUTER IS VALID */
    return OP_ENOSYS;
    // return OP_OK;
}