from rknn.api import RKNN
r=RKNN(verbose=True)
ret = r.config(
    target_platform='rk3568',
    mean_values=[[127.5, 127.5, 127.5]],
    std_values=[[127.5, 127.5, 127.5]],
    quantized_dtype='asymmetric_quantized-8',
    quantized_algorithm='normal',
    quantized_method='channel',
)
assert ret==0, ret
assert r.load_onnx(model='ppocrv5_mobile_rec_48x320.onnx')==0
assert r.build(do_quantization=True, dataset='dataset_ocr.txt')==0
assert r.export_rknn('models_rknn/ppocrv5_mobile_rec_48x320_int8.rknn')==0
r.release()
print('OCR INT8 OK')
