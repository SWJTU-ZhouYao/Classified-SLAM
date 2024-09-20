# A C++ example of <a href="https://github.com/ultralytics/yolov5" title="YOLOv5-seg">YOLOv5-seg</a> model.

## Prerequisites
We have tested this example on the Ubuntu 18.04, but it should be easy to compile in other platforms, since only OpenCV and TRT are needed. The following are the specific versions of the libraries used in our tested platform and the main requirements
<ol>
<li>CUDA: 11.5, cuDNN: 8.3.3</li>
<li>TensorRT-8.4.3.1, matched with your cuda version, refer to <a href="https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html" title="install guild">install guild</a>. Strongly recommend compiling from the source file, assuming you will be using multiple trt versions for different cuda.</li>
<li>OpenCV 4.5.5, refer to <a href="https://github.com/opencv/opencv/tree/4.5.5?tab=readme-ov-file" title="install guild">install guild</a>.</li>
<li>Note: You can download the trained torch model from the YOLOv5 repository. But if you want to use TensorRT to improve the inference speed in a C++ environment, as in this example, you need to use the export.py file provided in YOLOv5 source export the model to TensorRT. Although a TensorRT export is provided in this example, it may not match your CUDA version. And we provided an image from <a href="https://www.cvlibs.net/datasets/kitti/eval_odometry.php" title="kitti">kitti odometry dataset</a> for this testing.</li>
</ol>