#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnx.pb-c.h"
#include "utils.h"
#include "trace.h"
#include "test/test_utils.h"

/* TODO: Include statically linked library*/

int main()
{
  /* Not working yet. Makefile need some love */
  
  Onnx__ModelProto *model = openOnnxFile("./test/mnist/model.onnx");
  if (model == NULL)
  {
    perror("Error when opening the onnx file\n");
    exit(-1);
  }

  /* TODO: Run some inference on MNIST examples */
  Onnx__TensorProto *inp0set0 = openTensorProtoFile("./test/mnist/test_data_set_0/input_0.pb");
  Onnx__TensorProto *out0set0 = openTensorProtoFile("./test/mnist/test_data_set_0/output_0.pb");

  //Debug_PrintModelInformation(model);
  convertRawDataOfTensorProto(inp0set0);
  convertRawDataOfTensorProto(out0set0);

  // TODO Dirty trick. I expected the input name to be included in the
  // input_0, but apparently it is not. Dont know if memory for the name
  // is allocated... but it doesnt crash
  inp0set0->name = "Input3";
  printf("%s\n\n", inp0set0->name);

  Onnx__TensorProto *inputs[] = { inp0set0 };
  resolve(model, inputs, 1);
  Onnx__TensorProto **output = inference(model, inputs, 1);

  /* 11 is hardcoded, which is Plus214_Output_0 */
  compareAlmostEqualTensorProto(*output, out0set0);
  free(inp0set0);
  free(out0set0);
  free(model);

  /* TODO: Free all resources */

  return 0;
}
