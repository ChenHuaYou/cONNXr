#ifndef OPERATOR_STUB_H
#define OPERATOR_STUB_H

#include "operators/operator.h"
#include "operators/operator_info.h"

extern operator_info operator_stub_info;

operator_status operator_stub(Onnx__NodeProto *ctx);

operator_executer operator_stub_resolver(Onnx__NodeProto *ctx);

#endif
