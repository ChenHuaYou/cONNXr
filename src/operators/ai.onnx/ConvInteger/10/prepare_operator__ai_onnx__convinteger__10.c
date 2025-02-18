//this file was generated by ../../../../../../scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__convinteger__10.h"
#include "tracing.h"
#include "utils.h"

operator_status
prepare_operator__ai_onnx__convinteger__10(
    Onnx__NodeProto *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    //Onnx__TensorProto *i_x = searchInputByIndex(ctx, 0);
    //Onnx__TensorProto *i_w = searchInputByIndex(ctx, 1);
    //Onnx__TensorProto *i_x_zero_point = searchInputByIndex(ctx, 2);
    //Onnx__TensorProto *i_w_zero_point = searchInputByIndex(ctx, 3);

    // TRACE_TENSOR(2, true, i_x);
    // TRACE_TENSOR(2, true, i_w);
    // TRACE_TENSOR(2, x_zero_point, i_x_zero_point);
    // TRACE_TENSOR(2, w_zero_point, i_w_zero_point);

    // Onnx__AttributeProto *a_auto_pad = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"auto_pad");
    // Onnx__AttributeProto *a_dilations = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"dilations");
    // Onnx__AttributeProto *a_group = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"group");
    // Onnx__AttributeProto *a_kernel_shape = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"kernel_shape");
    // Onnx__AttributeProto *a_pads = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"pads");
    // Onnx__AttributeProto *a_strides = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"strides");

    // TRACE_ATTRIBUTE(2, a_auto_pad, a_auto_pad);
    // TRACE_ATTRIBUTE(2, a_dilations, a_dilations);
    // TRACE_ATTRIBUTE(2, a_group, a_group);
    // TRACE_ATTRIBUTE(2, a_kernel_shape, a_kernel_shape);
    // TRACE_ATTRIBUTE(2, a_pads, a_pads);
    // TRACE_ATTRIBUTE(2, a_strides, a_strides);

    Onnx__TensorProto *o_y = searchOutputByIndex(ctx, 0);

    /* ALLOCATE AND INITIALIZE CONTEXT HERE IF NEEDED */

    // char* default_auto_pad = ;
    // size_t default_n_dilations = ;
    // int64_t* default_dilations = ;
    // int64_t default_group = ;
    // size_t default_n_kernel_shape = ;
    // int64_t* default_kernel_shape = ;
    // size_t default_n_pads = ;
    // int64_t* default_pads = ;
    // size_t default_n_strides = ;
    // int64_t* default_strides = ;

    // context_operator__ai_onnx__convinteger__10 *op_ctx = NULL;
    // op_ctx = malloc(sizeof(context_operator__ai_onnx__convinteger__10));
    // TRACE_FATAL(0 , !op_ctx, "could not allocate executer_context");

    // op_ctx->auto_pad = a_auto_pad?strndup((char*)a_auto_pad->s.data, a_auto_pad->s.len):default_auto_pad;
    // op_ctx->n_dilations = a_dilations?a_dilations->n_ints:default_n_dilations;
    // op_ctx->dilations = a_dilations?a_dilations->ints:ARRAYDUP(default_dilations,default_n_dilations);
    // TRACE_FATAL(0, !op_ctx->dilations, "malloc failed");
    // op_ctx->group = a_group?a_group->i:default_group;
    // op_ctx->n_kernel_shape = a_kernel_shape?a_kernel_shape->n_ints:default_n_kernel_shape;
    // op_ctx->kernel_shape = a_kernel_shape?a_kernel_shape->ints:ARRAYDUP(default_kernel_shape,default_n_kernel_shape);
    // TRACE_FATAL(0, !op_ctx->kernel_shape, "malloc failed");
    // op_ctx->n_pads = a_pads?a_pads->n_ints:default_n_pads;
    // op_ctx->pads = a_pads?a_pads->ints:ARRAYDUP(default_pads,default_n_pads);
    // TRACE_FATAL(0, !op_ctx->pads, "malloc failed");
    // op_ctx->n_strides = a_strides?a_strides->n_ints:default_n_strides;
    // op_ctx->strides = a_strides?a_strides->ints:ARRAYDUP(default_strides,default_n_strides);
    // TRACE_FATAL(0, !op_ctx->strides, "malloc failed");

    // TRACE_VAR(2, true, op_ctx->auto_pad, "\"%s\"");
    // TRACE_ARRAY(2, true, op_ctx->dilations, , op_ctx->n_dilations, "%" PRId64);
    // TRACE_VAR(2, true, op_ctx->group, "%" PRId64);
    // TRACE_ARRAY(2, true, op_ctx->kernel_shape, , op_ctx->n_kernel_shape, "%" PRId64);
    // TRACE_ARRAY(2, true, op_ctx->pads, , op_ctx->n_pads, "%" PRId64);
    // TRACE_ARRAY(2, true, op_ctx->strides, , op_ctx->n_strides, "%" PRId64);

    /* INITIALIZE OUTPUTS DATA_TYPE AND SHAPE HERE */


    /* MALLOC OUTPUT TENSORS HERE */

    mallocTensorData(o_y);

    // TRACE_TENSOR(2, true, o_y);

    /* CHOOSE EXECUTER AND CONTEXT HERE */
    /* YOU MAY USE THE GENERATED RESOLVER */

    ctx->executer = execute_operator__ai_onnx__convinteger__10;
    // ctx->executer_context = op_ctx;

    TRACE_EXIT(1);

    /* CHANGE RETURN CODE IF THIS PREPARER IS VALID */
    return OP_ENOSYS;
    // return OP_OK;
}