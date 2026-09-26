import onnx
from onnx import helper, TensorProto
src = 'best.onnx'
dst = 'best_split.onnx'
m=onnx.load(src)
# 保留原计算图，仅将最终混合输出替换为拼接前的两个张量。
m.graph.ClearField('output')
m.graph.output.extend([
    helper.make_tensor_value_info('/model.23/Mul_2_output_0', TensorProto.FLOAT, [1,4,8400]),
    helper.make_tensor_value_info('/model.23/Sigmoid_output_0', TensorProto.FLOAT, [1,1,8400]),
])
onnx.checker.check_model(m)
onnx.save(m,dst)
print('saved',dst)
