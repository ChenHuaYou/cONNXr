#ifndef OPERATOR_CHECK_H
#define OPERATOR_CHECK_H

#include "operators/operator.h"
#include "operators/operator_info.h"
#include "stdbool.h"

bool
operator_check(Onnx__NodeProto *ctx, operator_info *info);

bool
operator_check_range(Onnx__NodeProto *ctx, operator_info *info);

bool
operator_check_attributes(Onnx__NodeProto *ctx, operator_info *info);

bool
operator_check_tensors(Onnx__NodeProto *ctx, operator_info *info);

bool
operator_check_constraint(Onnx__NodeProto *ctx, operator_info *info);

#endif