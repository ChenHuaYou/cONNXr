#ifndef INFERENCE_H
#define INFERENCE_H
#include "onnx.pb-c.h"
#include "operators/operator.h"

Onnx__TensorProto** inference(Onnx__ModelProto *model,
                              Onnx__TensorProto **inputs);

void resolve(Onnx__ModelProto *model,
             Onnx__TensorProto **inputs,
             int nInputs);

/* TODO Shouldn't be global */
#define MAX_NUM_OF_NODES 200

#endif
