//
// Created by zy on 24-9-20.
//
#include <vector>
#include <list>
#include <string>
#include <iostream>
#include <cmath>
#include <fstream>
#include <boost/format.hpp>
using namespace std;
// for cv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
using namespace cv;
// for cuda and trt
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "NvInferPlugin.h"
using namespace nvinfer1;
#include "logging.h"
#include "utils.h"

struct OutputSeg {
    int id;
    float confidence;
    Rect box;
    Mat boxMask;
};

void Segmentation_from_YoloV5_by_TRT(IExecutionContext& context, Mat& frame, vector<OutputSeg>& outputs);
void doInference(IExecutionContext& context, float* input, float* output, float* output1, int batchSize);

int main(int argc, char **argv) {
    // 读取模型
    char* trtModelStream{ nullptr };
    size_t size{0};
    std::ifstream file("/media/vigi/Elements/YOLOv5-seg/YOLOv5/Yolov5-instance-seg-tensorrt/models/yolov5s-seg.engine", std::ios::binary);
    if (file.good()) {
        std::cout<<"load engine success"<<std::endl;
        file.seekg(0, file.end);
        size = file.tellg();

        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);//
        file.read(trtModelStream, size);
        file.close();\
    }
    else {
        std::cout << "load engine failed" << std::endl;
        return 1;
    }
    Logger gLogger;
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    const string img_path = "/home/vigi/Third_Paper/YOLOv5-seg-example/000000.png";
    Mat frame = imread(img_path, IMREAD_COLOR);

    vector<OutputSeg> outputs;
    Segmentation_from_YoloV5_by_TRT(*context, frame, outputs);
    Mat mask = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
    for(const auto& i : outputs){
        mask(i.box).setTo(1, i.boxMask);
    }
    Mat mask_frame = frame.clone();
    Scalar color = Scalar(151, 255, 255);
    Mat coloredRoi = (0.3 * color + 0.7 * mask_frame); // add at last dim
    coloredRoi.convertTo(coloredRoi, CV_8UC3);
    vector<Mat> contours;
    Mat hierarchy;
    findContours(mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    drawContours(coloredRoi, contours, -1, color, 5, LINE_8, hierarchy, 100);
    coloredRoi.copyTo(mask_frame, mask);
    imshow("Segmentation Results", mask_frame);
    waitKey(0);


    return 0;
}

void Segmentation_from_YoloV5_by_TRT(IExecutionContext& context, Mat& frame, vector<OutputSeg>& outputs){
    static float data[3 * 640 * 640];
    Mat pr_img0, pr_img;
    vector<int> padsize;
    pr_img=preprocess_img(frame, 640, 640, padsize); // resize frame for input
    int i = 0;
    for (int row = 0; row < 640; ++row) {
        uchar* uc_pixel = pr_img.data + row * pr_img.step;//pr_img.step=widthx3
        for (int col = 0; col < 640; ++col)
        {
            data[i] = (float)uc_pixel[2] / 255.0;
            data[i + 640 * 640] = (float)uc_pixel[1] / 255.0;
            data[i + 2 * 640 * 640] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    } // data

    static float prob[2948400];
    static float prob1[819200];

    doInference(context, data, prob, prob1, 1);

    vector<int> classIds;
    vector<float> confidences;
    vector<cv::Rect> boxes;
    vector<vector<float>> picked_proposals;
    int newh = padsize[0], neww = padsize[1], padh = padsize[2], padw = padsize[3];
    float ratio_h = (float)frame.rows / newh;
    float ratio_w = (float)frame.cols / neww;
    int net_width = 117;
    float* pdata = prob;
    for (int j = 0; j < 25200; ++j) {
        float box_score = pdata[4];
        if (box_score >= 0.1) {
            cv::Mat scores(1, 80, CV_32FC1, pdata + 5);
            Point classIdPoint;
            double max_class_socre;
            minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
            max_class_socre = (float)max_class_socre;
            if (max_class_socre >= 0.1) {
                std::vector<float> temp_proto(pdata + 5 + 80, pdata + net_width);
                picked_proposals.push_back(temp_proto);
                float x = (pdata[0] - padw) * ratio_w;  //x
                float y = (pdata[1] - padh) * ratio_h;  //y
                float w = pdata[2] * ratio_w;  //w
                float h = pdata[3] * ratio_h;  //h
                int left = MAX((x - 0.5 * w) , 0);
                int top = MAX((y - 0.5 * h) , 0);
                classIds.push_back(classIdPoint.x); //
                confidences.push_back(max_class_socre * box_score);
                boxes.push_back(Rect(left, top, int(w ), int(h )));
            }
        }
        pdata += net_width;
    }

    vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, 0.1, 0.5, nms_result);
    vector<vector<float>> temp_mask_proposals;
    Rect holeImgRect(0, 0, frame.cols, frame.rows);
    vector<OutputSeg> output;
    for (int idx : nms_result) {
        OutputSeg result;
        result.id = classIds[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx] & holeImgRect;
        output.push_back(result);
        temp_mask_proposals.push_back(picked_proposals[idx]);
    }
    Mat maskProposals;
    for (const auto & temp_mask_proposal : temp_mask_proposals)
        maskProposals.push_back( Mat(temp_mask_proposal).t() );
    pdata = prob1;
    std::vector<float> mask(pdata, pdata + 819200); // 32 * 160 * 160
    Mat mask_protos = Mat(mask);
    Mat protos = mask_protos.reshape(0, { 32, 25600 }); //160 * 160
    Mat matmulRes = (maskProposals * protos).t();
    Mat masks = matmulRes.reshape(output.size(), { 160, 160 });

    vector<Mat> maskChannels;
    split(masks, maskChannels);
    for (int k = 0; k < output.size(); ++k) {
        Mat dest, mask;
        cv::exp(-maskChannels[k], dest);
        dest = 1.0 / (1.0 + dest);//160*160
        Rect roi(int((float)padw * 0.25), int((float)padh * 0.25), int(160 - padw / 2), int(160 - padh / 2));
        dest = dest(roi);
        resize(dest, mask, frame.size(), INTER_NEAREST);
        Rect temp_rect = output[k].box;
        mask = mask(temp_rect) > 0.5;
        output[k].boxMask = mask;
    }

    outputs = output;
}

void doInference(IExecutionContext& context, float* input, float* output, float* output1, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 3);
    void* buffers[3];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex("images");
    const int outputIndex = engine.getBindingIndex("output0");
    const int outputIndex1 = engine.getBindingIndex("output1");

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * 640 * 640 * sizeof(float)));//
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * 2948400 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * 819200 * sizeof(float)));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * 640 * 640 * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueueV2(buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * 2948400 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(output1, buffers[outputIndex1], batchSize * 819200 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    CHECK(cudaFree(buffers[outputIndex1]));
}
