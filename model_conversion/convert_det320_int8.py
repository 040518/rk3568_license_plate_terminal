from rknn.api import RKNN
r=RKNN(verbose=True)
assert r.config(
    target_platform='rk3568',
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    quantized_dtype='asymmetric_quantized-8',
    quantized_algorithm='normal',
    quantized_method='channel',
) == 0
assert r.load_onnx(model='best_320_split.onnx')==0
assert r.build(do_quantization=True,dataset='dataset.txt')==0
assert r.export_rknn('models_rknn/best_320_int8_split.rknn')==0
r.release()
print('DET320 INT8 OK')
