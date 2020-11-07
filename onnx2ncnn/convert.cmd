mkdir srmd_ncnn_models
onnx2ncnn ../onnx_models/srmd_x2-sim.onnx srmd_ncnn_models/srmd_x2.param srmd_ncnn_models/srmd_x2.bin
echo "Convert srmd_x2 success!"
onnx2ncnn ../onnx_models/srmd_x3-sim.onnx srmd_ncnn_models/srmd_x3.param srmd_ncnn_models/srmd_x3.bin
echo "Convert srmd_x3 success!"
onnx2ncnn ../onnx_models/srmd_x4-sim.onnx srmd_ncnn_models/srmd_x4.param srmd_ncnn_models/srmd_x4.bin
echo "Convert srmd_x4 success!"
onnx2ncnn ../onnx_models/srmdnf_x2-sim.onnx srmd_ncnn_models/srmdnf_x2.param srmd_ncnn_models/srmdnf_x2.bin
echo "Convert srmdnf_x2 success!"
onnx2ncnn ../onnx_models/srmdnf_x3-sim.onnx srmd_ncnn_models/srmdnf_x3.param srmd_ncnn_models/srmdnf_x3.bin
echo "Convert srmdnf_x3 success!"
onnx2ncnn ../onnx_models/srmdnf_x4-sim.onnx srmd_ncnn_models/srmdnf_x4.param srmd_ncnn_models/srmdnf_x4.bin
echo "Convert srmdnf_x4 success!"
echo "Convert Complete!"