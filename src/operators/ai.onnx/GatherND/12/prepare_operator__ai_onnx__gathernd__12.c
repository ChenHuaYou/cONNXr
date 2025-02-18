//this file was generated by ../../../../../../scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__gathernd__12.h"
#include "tracing.h"
#include "utils.h"

operator_status
prepare_operator__ai_onnx__gathernd__12(
    Onnx__NodeProto *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    //Onnx__TensorProto *i_data = searchInputByIndex(ctx, 0);
    //Onnx__TensorProto *i_indices = searchInputByIndex(ctx, 1);

    // TRACE_TENSOR(2, true, i_data);
    // TRACE_TENSOR(2, true, i_indices);

    // Onnx__AttributeProto *a_batch_dims = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"batch_dims");

    // TRACE_ATTRIBUTE(2, a_batch_dims, a_batch_dims);

    Onnx__TensorProto *o_output = searchOutputByIndex(ctx, 0);

    /* ALLOCATE AND INITIALIZE CONTEXT HERE IF NEEDED */

    // int64_t default_batch_dims = ;

    // context_operator__ai_onnx__gathernd__12 *op_ctx = NULL;
    // op_ctx = malloc(sizeof(context_operator__ai_onnx__gathernd__12));
    // TRACE_FATAL(0 , !op_ctx, "could not allocate executer_context");

    // op_ctx->batch_dims = a_batch_dims?a_batch_dims->i:default_batch_dims;

    // TRACE_VAR(2, true, op_ctx->batch_dims, "%" PRId64);

    /* INITIALIZE OUTPUTS DATA_TYPE AND SHAPE HERE */


    /* MALLOC OUTPUT TENSORS HERE */

    mallocTensorData(o_output);

    // TRACE_TENSOR(2, true, o_output);

    /* CHOOSE EXECUTER AND CONTEXT HERE */
    /* YOU MAY USE THE GENERATED RESOLVER */

    ctx->executer = execute_operator__ai_onnx__gathernd__12;
    // ctx->executer_context = op_ctx;

    TRACE_EXIT(1);

    /* CHANGE RETURN CODE IF THIS PREPARER IS VALID */
    return OP_ENOSYS;
    // return OP_OK;
}