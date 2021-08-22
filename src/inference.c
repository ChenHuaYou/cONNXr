#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnx.pb-c.h"
#include "operators/operator.h"
#include "utils.h"
#include "tracing.h"
#include "inference.h"
#include "operators/operator_set.h"

// Won't be global in the future

void resolveModel(Onnx__ModelProto *model)
{
    TRACE_ENTRY(1);
    /* Resolving operators and input/outputs. Has to be moved outside of infeference */

    TRACE_FATAL(0, model->graph->n_node > MAX_NUM_OF_NODES, "The number of nodes of the model is greater than the hardcoded one");

    for (int nodeIdx = 0; nodeIdx < model->graph->n_node; nodeIdx++)
    {

        // Search the inputs for a node
        model->graph->node[nodeIdx]->inputs = malloc(sizeof(Onnx__TensorProto *) * model->graph->node[nodeIdx]->n_input);
        for (int i = 0; i < model->graph->node[nodeIdx]->n_input; i++)
        {
            model->graph->node[nodeIdx]->inputs[i] = searchTensorProtoByName(model, model->graph->node[nodeIdx]->input[i]);
            if (model->graph->node[nodeIdx]->inputs[i] && model->graph->node[nodeIdx]->inputs[i]->has_raw_data){
                /* If the tensor has raw data, deserialize it */
                TRACE(1, true, "input %s has raw data", model->graph->node[nodeIdx]->input[i]);
                // TODO: Not tested. Crashing but currently not needed
                convertRawDataOfTensorProto(model->graph->node[nodeIdx]->inputs[i]);
            }
        }

        // Allocate memory for future outputs and set the name
        model->graph->node[nodeIdx]->outputs = malloc(sizeof(Onnx__TensorProto *) * model->graph->node[nodeIdx]->n_output);
        for (int i = 0; i < model->graph->node[nodeIdx]->n_output; i++)
        {
            model->graph->node[nodeIdx]->outputs[i] = malloc(sizeof(Onnx__TensorProto));
            init_tensor_proto(model->graph->node[nodeIdx]->outputs[i]);
            model->graph->node[nodeIdx]->outputs[i]->name = strdup(model->graph->node[nodeIdx]->output[i]);

            // TODO This is unset at this point but set afterward inside each
            // function. However there is a problem because some node output
            // is some node else input. Hence if the type is unset it can't
            // be resolved. Hardcoded to FLOAT but this is a HUGE TODO
            model->graph->node[nodeIdx]->outputs[i]->data_type = 1;
        }

        /*** Prototyping ***/
        // Check model->opset_import->has_version must be True
        // More than 1 opset can be imported. Iterate n_opset_import
        // model->opset_import[0]->version
        // TODO Hackish temporal solution. Use opset 12.
        size_t version = 12;
        operator_preparer prepare = operator_set_find_preparer(model->graph->node[nodeIdx]->op_type, version);
        TRACE_FATAL(0, !prepare, "No prepare function could be found for operator '%s' version '%zu'", model->graph->node[nodeIdx]->op_type, version);
        prepare(model->graph->node[nodeIdx]);
    }
    TRACE_EXIT(1);
}

Onnx__TensorProto** inference(Onnx__ModelProto *model, Onnx__TensorProto **inputs)
{
    TRACE_ENTRY(1);
    TRACE(1, true, "The graph has nodes=%zu", model->graph->n_node);

    /* Run inference */
    for (int nodeIdx = 0; nodeIdx < model->graph->n_node; nodeIdx++)
    {
        TRACE(1, true, "Running node %d, operator=%s", nodeIdx, model->graph->node[nodeIdx]->op_type);
        model->graph->node[nodeIdx]->executer(model->graph->node[nodeIdx]);
    }

    // TODO
    TRACE_EXIT(1);
    //freeContext(all_context, model);
    return model->graph->node[model->graph->n_node-1]->outputs;
}

