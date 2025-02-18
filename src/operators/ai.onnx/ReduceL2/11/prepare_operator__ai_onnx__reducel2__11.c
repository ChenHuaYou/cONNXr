//this file was generated by ../../../../../../scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__reducel2__11.h"
#include "tracing.h"
#include "utils.h"

operator_status
prepare_operator__ai_onnx__reducel2__11(
    Onnx__NodeProto *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    //Onnx__TensorProto *i_data = searchInputByIndex(ctx, 0);

    // TRACE_TENSOR(2, true, i_data);

    // Onnx__AttributeProto *a_axes = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"axes");
    // Onnx__AttributeProto *a_keepdims = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"keepdims");

    // TRACE_ATTRIBUTE(2, a_axes, a_axes);
    // TRACE_ATTRIBUTE(2, a_keepdims, a_keepdims);

    Onnx__TensorProto *o_reduced = searchOutputByIndex(ctx, 0);

    /* ALLOCATE AND INITIALIZE CONTEXT HERE IF NEEDED */

    // size_t default_n_axes = ;
    // int64_t* default_axes = ;
    // int64_t default_keepdims = ;

    // context_operator__ai_onnx__reducel2__11 *op_ctx = NULL;
    // op_ctx = malloc(sizeof(context_operator__ai_onnx__reducel2__11));
    // TRACE_FATAL(0 , !op_ctx, "could not allocate executer_context");

    // op_ctx->n_axes = a_axes?a_axes->n_ints:default_n_axes;
    // op_ctx->axes = a_axes?a_axes->ints:ARRAYDUP(default_axes,default_n_axes);
    // TRACE_FATAL(0, !op_ctx->axes, "malloc failed");
    // op_ctx->keepdims = a_keepdims?a_keepdims->i:default_keepdims;

    // TRACE_ARRAY(2, true, op_ctx->axes, , op_ctx->n_axes, "%" PRId64);
    // TRACE_VAR(2, true, op_ctx->keepdims, "%" PRId64);

    /* INITIALIZE OUTPUTS DATA_TYPE AND SHAPE HERE */


    /* MALLOC OUTPUT TENSORS HERE */

    mallocTensorData(o_reduced);

    // TRACE_TENSOR(2, true, o_reduced);

    /* CHOOSE EXECUTER AND CONTEXT HERE */
    /* YOU MAY USE THE GENERATED RESOLVER */

    ctx->executer = execute_operator__ai_onnx__reducel2__11;
    // ctx->executer_context = op_ctx;

    TRACE_EXIT(1);

    /* CHANGE RETURN CODE IF THIS PREPARER IS VALID */
    return OP_ENOSYS;
    // return OP_OK;
}