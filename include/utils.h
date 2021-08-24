#ifndef UTILS_H
#define UTILS_H
#include "onnx.pb-c.h"
#include "inference.h"
#include "trace.h"

/* Max size for strings */
#define MAX_CHAR_SIZE 50

void connectNodes(Onnx__ModelProto *model,int k_node, int k_input);
Onnx__AttributeProto* searchAttributeNyName(size_t n_attribute,
                                            Onnx__AttributeProto **attribute,
                                            char *name);
Onnx__TensorProto* searchInputByIndex(Onnx__NodeProto *ctx,
                                     int index);
Onnx__TensorProto* searchOutputByIndex(Onnx__NodeProto *ctx,
                                      int index);
Onnx__ModelProto* openOnnxFile(char *fname);
Onnx__TensorProto* openTensorProtoFile(char *fname);

size_t exportTensorProtoFile(const Onnx__TensorProto *tensor, char *fname);

int convertRawDataOfTensorProto(Onnx__TensorProto *tensor);

void mallocTensorProto(Onnx__TensorProto *tp,
                       Onnx__TensorProto__DataType data_type,
                       size_t n_dims,
                       size_t n_data);

void init_tensor_proto(Onnx__TensorProto *tp);

size_t strnlen(const char *src, size_t length);
char*  strndup(const char *src, size_t length);
char*  strdup(const char *src);
void*  memdup(const void *src, size_t size);
#define ARRAYDUP(SRC, LENGTH) memdup(SRC, sizeof((SRC)[0])*(LENGTH))

void* mallocTensorData(Onnx__TensorProto *dst);
void* freeTensorData(Onnx__TensorProto *dst);


bool tensorCheckBroadcasting(Onnx__TensorProto *src, Onnx__TensorProto *dst);
void tensorIdxToSubscript(Onnx__TensorProto *x, int *subscript, int idx);
int tensorSubscriptToIdx(Onnx__TensorProto *x,int *subscript);

#define tensorAdd(type,o_C,i_A,i_B)\
    do{\
        if(!tensorCheckBroadcasting(i_A,i_B)){\
            TRACE_LEVEL0("invalid broadcasting");\
            exit(EXIT_FAILURE);\
        }else{\
            int *subscript = malloc(o_C->n_dims*sizeof(int));\
            for(int i=0; i<o_C->n_##type##_data; i++){\
                tensorIdxToSubscript(o_C, subscript, i);\
                o_C->type##_data[i] = i_A->type##_data[tensorSubscriptToIdx(i_A,subscript)]\
                    + i_B->type##_data[tensorSubscriptToIdx(i_B,subscript)];\
            }\
            free(subscript);\
        }\
    }while(0)


#endif
