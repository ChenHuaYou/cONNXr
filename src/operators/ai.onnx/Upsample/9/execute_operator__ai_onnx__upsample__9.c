//this file was generated by ../../../../../../scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__upsample__9.h"
#include "tracing.h"
#include "utils.h"

operator_status
execute_operator__ai_onnx__upsample__9(
    Onnx__NodeProto *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    //Onnx__TensorProto *i_X = searchInputByIndex(ctx, 0);
    //Onnx__TensorProto *i_scales = searchInputByIndex(ctx, 1);

    // TRACE_TENSOR(2, true, i_X);
    // TRACE_TENSOR(2, true, i_scales);

    // context_operator__ai_onnx__upsample__9 *op_ctx = ctx->executer_context;

    // char* mode = op_ctx->mode;

    // TRACE_VAR(2, true, mode, "\"%s\"");

    //Onnx__TensorProto *o_Y = searchOutputByIndex(ctx, 0);

    // TRACE_TENSOR(2, true, o_Y);

    /* DO CALCULATION HERE */


    TRACE_EXIT(1);

    /* CHANGE RETURN CODE IF THIS EXECUTER IS VALID */
    return OP_ENOSYS;
    // return OP_OK;
}