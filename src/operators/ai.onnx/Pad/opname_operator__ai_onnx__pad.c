//this file was generated by ../../../../../scripts/onnx_generator/OperatorSets.py

#include "operators/operator_set.h"

extern operator_set_opversion opversion_operator__ai_onnx__pad__1;
extern operator_set_opversion opversion_operator__ai_onnx__pad__2;
extern operator_set_opversion opversion_operator__ai_onnx__pad__11;

operator_set_opname opname_operator__ai_onnx__pad = {
    .name = "Pad",
    .opversions = {
        &opversion_operator__ai_onnx__pad__1,
        &opversion_operator__ai_onnx__pad__2,
        &opversion_operator__ai_onnx__pad__11,
        NULL
    }
};