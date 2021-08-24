import onnx
import numpy as np
from onnx.backend.test.case.node import expect


node = onnx.helper.make_node(
    'Abs',
    inputs=['x'],
    outputs=['y'],
)
x = np.random.randn(3, 4, 5).astype(np.float32)
y = -abs(x)

res = expect(node, inputs=[x], outputs=[y],
       name='test_abs')

