//this file was generated by ../../../../../scripts/onnx_generator/OperatorSets.py

#include "operators/operator_set.h"

extern operator_set_opversion opversion_operator__ai_onnx__gathernd__11;
extern operator_set_opversion opversion_operator__ai_onnx__gathernd__12;

operator_set_opname opname_operator__ai_onnx__gathernd = {
    .name = "GatherND",
    .opversions = {
        &opversion_operator__ai_onnx__gathernd__11,
        &opversion_operator__ai_onnx__gathernd__12,
        NULL
    }
};