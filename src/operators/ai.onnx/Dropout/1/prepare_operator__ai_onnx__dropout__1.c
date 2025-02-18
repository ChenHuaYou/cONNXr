//this file was generated by ../../../../../../scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__dropout__1.h"
#include "tracing.h"
#include "utils.h"

operator_status
prepare_operator__ai_onnx__dropout__1(
    Onnx__NodeProto *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    //Onnx__TensorProto *i_data = searchInputByIndex(ctx, 0);

    // TRACE_TENSOR(2, true, i_data);

    // Onnx__AttributeProto *a_consumed_inputs = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"consumed_inputs");
    // Onnx__AttributeProto *a_is_test = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"is_test");
    // Onnx__AttributeProto *a_ratio = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"ratio");

    // TRACE_ATTRIBUTE(2, a_consumed_inputs, a_consumed_inputs);
    // TRACE_ATTRIBUTE(2, a_is_test, a_is_test);
    // TRACE_ATTRIBUTE(2, a_ratio, a_ratio);

    Onnx__TensorProto *o_output = searchOutputByIndex(ctx, 0);
    Onnx__TensorProto *o_mask = searchOutputByIndex(ctx, 1);

    /* ALLOCATE AND INITIALIZE CONTEXT HERE IF NEEDED */

    // size_t default_n_consumed_inputs = ;
    // int64_t* default_consumed_inputs = ;
    // int64_t default_is_test = ;
    // float default_ratio = ;

    // context_operator__ai_onnx__dropout__1 *op_ctx = NULL;
    // op_ctx = malloc(sizeof(context_operator__ai_onnx__dropout__1));
    // TRACE_FATAL(0 , !op_ctx, "could not allocate executer_context");

    // op_ctx->n_consumed_inputs = a_consumed_inputs?a_consumed_inputs->n_ints:default_n_consumed_inputs;
    // op_ctx->consumed_inputs = a_consumed_inputs?a_consumed_inputs->ints:ARRAYDUP(default_consumed_inputs,default_n_consumed_inputs);
    // TRACE_FATAL(0, !op_ctx->consumed_inputs, "malloc failed");
    // op_ctx->is_test = a_is_test?a_is_test->i:default_is_test;
    // op_ctx->ratio = a_ratio?a_ratio->f:default_ratio;

    // TRACE_ARRAY(2, true, op_ctx->consumed_inputs, , op_ctx->n_consumed_inputs, "%" PRId64);
    // TRACE_VAR(2, true, op_ctx->is_test, "%" PRId64);
    // TRACE_VAR(2, true, op_ctx->ratio, "%f");

    /* INITIALIZE OUTPUTS DATA_TYPE AND SHAPE HERE */


    /* MALLOC OUTPUT TENSORS HERE */

    mallocTensorData(o_output);
    mallocTensorData(o_mask);

    // TRACE_TENSOR(2, true, o_output);
    // TRACE_TENSOR(2, mask, o_mask);

    /* CHOOSE EXECUTER AND CONTEXT HERE */
    /* YOU MAY USE THE GENERATED RESOLVER */

    ctx->executer = execute_operator__ai_onnx__dropout__1;
    // ctx->executer_context = op_ctx;

    TRACE_EXIT(1);

    /* CHANGE RETURN CODE IF THIS PREPARER IS VALID */
    return OP_ENOSYS;
    // return OP_OK;
}