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
//
#define NODE model->graph->node

void resolve(Onnx__ModelProto *model)
{
    TRACE_ENTRY(1);
    /* Resolving operators and input/outputs. Has to be moved outside of infeference */

    TRACE_FATAL(0, model->graph->n_node > MAX_NUM_OF_NODES, "The number of nodes of the model is greater than the hardcoded one");
    model->graph->inputs = malloc(sizeof(Onnx__TensorProto **) * model->graph->n_input);

    for (int nodeIdx = 0; nodeIdx < model->graph->n_node; nodeIdx++){
        //printf("node: %s\n",NODE[nodeIdx]->name);
        // Allocate memory for future outputs and set the name
        model->graph->node[nodeIdx]->outputs = malloc(sizeof(Onnx__TensorProto *) * model->graph->node[nodeIdx]->n_output);
        model->graph->node[nodeIdx]->inputs = malloc(sizeof(Onnx__TensorProto *) * model->graph->node[nodeIdx]->n_input);
        for (int i = 0; i < model->graph->node[nodeIdx]->n_output; i++){
            //printf("output: %s\n",NODE[nodeIdx]->output[i]);
            model->graph->node[nodeIdx]->outputs[i] = malloc(sizeof(Onnx__TensorProto));
            init_tensor_proto(model->graph->node[nodeIdx]->outputs[i]);
            model->graph->node[nodeIdx]->outputs[i]->name = strdup(model->graph->node[nodeIdx]->output[i]);
            bool fuck = true;
            // match from model->graph->output
            for(int j=0; j<model->graph->n_output; j++){
                //printf("grap_output: %s\n", model->graph->output[j]->name);
                if(!strcmp(model->graph->output[j]->name,model->graph->node[nodeIdx]->outputs[i]->name)){
                    fuck = false;
                    model->graph->node[nodeIdx]->outputs[i]->n_dims = model->graph->output[j]->type->tensor_type->shape->n_dim;
                    model->graph->node[nodeIdx]->outputs[i]->dims = malloc(sizeof(int64_t *)*model->graph->node[nodeIdx]->outputs[i]->n_dims);
                    for(int k=0; k<model->graph->node[nodeIdx]->outputs[i]->n_dims; k++){
                        model->graph->node[nodeIdx]->outputs[i]->dims[k] = model->graph->output[j]->type->tensor_type->shape->dim[k]->dim_value;
                        model->graph->node[nodeIdx]->outputs[i]->data_type = model->graph->output[j]->type->tensor_type->elem_type;
                    }
                }
            }
            // match from model->graph->value_info
            for(int j=0; j<model->graph->n_value_info; j++){
                //printf("valueinfo: %s\n", model->graph->value_info[j]->name);
                if(!strcmp(model->graph->value_info[j]->name,model->graph->node[nodeIdx]->outputs[i]->name)){
                    fuck = false;
                    model->graph->node[nodeIdx]->outputs[i]->n_dims = model->graph->value_info[j]->type->tensor_type->shape->n_dim;
                    model->graph->node[nodeIdx]->outputs[i]->dims = malloc(sizeof(int64_t *)*model->graph->node[nodeIdx]->outputs[i]->n_dims);
                    for(int k=0; k<model->graph->node[nodeIdx]->outputs[i]->n_dims; k++){
                        model->graph->node[nodeIdx]->outputs[i]->dims[k] = model->graph->value_info[j]->type->tensor_type->shape->dim[k]->dim_value;
                        model->graph->node[nodeIdx]->outputs[i]->data_type = model->graph->value_info[j]->type->tensor_type->elem_type;
                    }
                }
            }

            // TODO This is unset at this point but set afterward inside each
            // function. However there is a problem because some node output
            // is some node else input. Hence if the type is unset it can't
            // be resolved. Hardcoded to FLOAT but this is a HUGE TODO
            //model->graph->node[nodeIdx]->outputs[i]->data_type = 1;
        }

        // connectNodes
        for (int i = 0; i < model->graph->node[nodeIdx]->n_input; i++)
        {
            connectNodes(model, nodeIdx, i);
            if (model->graph->node[nodeIdx]->inputs[i] && model->graph->node[nodeIdx]->inputs[i]->has_raw_data){
                /* If the tensor has raw data, deserialize it */
                TRACE(1, true, "input %s has raw data", model->graph->node[nodeIdx]->input[i]);
                // TODO: Not tested. Crashing but currently not needed
                convertRawDataOfTensorProto(model->graph->node[nodeIdx]->inputs[i]);
            }
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
        //printf("prepare\n");
        checkNode(model->graph->node[nodeIdx]);
    }
    TRACE_EXIT(1);
}

Onnx__TensorProto** inference(Onnx__ModelProto *model, Onnx__TensorProto **inputs)
{
    if(!model->resolved){
        resolve(model);
    }
    int n_bind = 0;
    for(int i=0; i<model->graph->n_input; i++){
        for(int j=0; inputs[j]; j++){
            printf("compare input %s <=> %s \n", model->graph->input[i]->name, inputs[j]->name);
            if(!strcmp(model->graph->input[i]->name,inputs[j]->name)){
                *model->graph->inputs[i] = inputs[j];
                n_bind ++;
            }
        }
    }
    TRACE_ENTRY(1);
    TRACE(1, true, "The graph has nodes=%zu", model->graph->n_node);

    /* Run inference */
    for (int nodeIdx = 0; nodeIdx < model->graph->n_node; nodeIdx++)
    {
        TRACE(0, true, "Running node %d, operator=%s", nodeIdx, model->graph->node[nodeIdx]->op_type);
        model->graph->node[nodeIdx]->executer(model->graph->node[nodeIdx]);
    }

    // TODO
    TRACE_EXIT(1);
    //freeContext(all_context, model);
    return model->graph->node[model->graph->n_node-1]->outputs;
}

