//this file was generated by ../../../../../../scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__clip__1.h"
#include "tracing.h"
#include "utils.h"

operator_status
execute_operator__ai_onnx__clip__1(
    Onnx__NodeProto *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    //Onnx__TensorProto *i_input = searchInputByIndex(ctx, 0);

    // TRACE_TENSOR(2, true, i_input);

    // context_operator__ai_onnx__clip__1 *op_ctx = ctx->executer_context;

    // size_t n_consumed_inputs = op_ctx->n_consumed_inputs;
    // int64_t* consumed_inputs = op_ctx->consumed_inputs;
    // float max = op_ctx->max;
    // float min = op_ctx->min;

    // TRACE_ARRAY(2, true, consumed_inputs, , n_consumed_inputs, "%" PRId64);
    // TRACE_VAR(2, true, max, "%f");
    // TRACE_VAR(2, true, min, "%f");

    //Onnx__TensorProto *o_output = searchOutputByIndex(ctx, 0);

    // TRACE_TENSOR(2, true, o_output);

    /* DO CALCULATION HERE */


    TRACE_EXIT(1);

    /* CHANGE RETURN CODE IF THIS EXECUTER IS VALID */
    return OP_ENOSYS;
    // return OP_OK;
}