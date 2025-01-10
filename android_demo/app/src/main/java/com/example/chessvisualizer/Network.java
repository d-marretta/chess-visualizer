package com.example.chessvisualizer;

import static androidx.core.math.MathUtils.clamp;
import static org.opencv.core.Core.add;
import static org.opencv.core.Core.compare;
import static org.opencv.core.Core.divide;
import static org.opencv.core.Core.exp;
import static org.opencv.core.Core.gemm;
import static org.opencv.core.Core.multiply;
import static org.opencv.core.Core.split;
import static org.opencv.core.Core.transpose;
import static org.opencv.imgproc.Imgproc.INTER_LINEAR;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.util.Log;

import com.google.android.gms.tasks.Task;
import com.google.android.gms.tflite.client.TfLiteInitializationOptions;
import com.google.android.gms.tflite.java.TfLite;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.InterpreterApi;
import org.tensorflow.lite.gpu.GpuDelegateFactory;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import org.opencv.dnn.Dnn;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class Network {

    private static final int YOLO_HEIGHT = 640;
    private static final int YOLO_WIDTH = 640;
    private static final int SEG_CHANNELS = 32;
    private static final int SEG_W = 160;
    private static final int SEG_H = 160;
    private static final int TOP_K = 100;
    private static final float SEGMENTATION_THRESHOLD = 0.0f;
    private InterpreterApi seg_interpreter;
    private int segNumAttributes;
    private int segNumAnchors;
    private InterpreterApi det_interpreter;
    private ImageProcessor imageProcessor;

    public interface InitializationCallback {
        void onInitializationComplete();
        void onInitializationFailed(Exception e);
    }

    public static class Obj {
        public int label;
        public Rect rect;
        public float probability;
        public Mat boxMask;
    }

    public Network(Context context, String segModelName, String detModelName, InitializationCallback callback) throws IOException {
        Task<Void> initializeTask = TfLite.initialize(context,
                                        TfLiteInitializationOptions.builder()
                                                .setEnableGpuDelegateSupport(true)
                                                .build());


        MappedByteBuffer seg_model = loadModelFile(context, segModelName);
        MappedByteBuffer det_model = loadModelFile(context, detModelName);
        initializeTask.addOnSuccessListener(a -> {
                    imageProcessor = new ImageProcessor.Builder()
                            .add(new ResizeOp(YOLO_HEIGHT, YOLO_WIDTH, ResizeOp.ResizeMethod.BILINEAR))
                            .add(new NormalizeOp(0, 255.0f))
                            .build();

                    seg_interpreter = InterpreterApi.create(seg_model,
                            new InterpreterApi.Options().setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY)
                                                        .addDelegateFactory(new GpuDelegateFactory()));
                    det_interpreter = InterpreterApi.create(det_model,
                            new InterpreterApi.Options().setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY)
                                    .addDelegateFactory(new GpuDelegateFactory()));

                    setSegOutputDims();

                    callback.onInitializationComplete();
                })
                .addOnFailureListener(e -> {
                    Log.e("Interpreter", String.format("Cannot initialize interpreter: %s", e.getMessage()));
                    callback.onInitializationFailed(e);
                });


    }

    private void setSegOutputDims(){
        int[] output0Shape = seg_interpreter.getOutputTensor(0).shape(); // [1,37.8400]

        if (output0Shape != null) {
            segNumAttributes = output0Shape[1];
            segNumAnchors = output0Shape[2];
        }

    }

    private MappedByteBuffer loadModelFile(Context context, String modelName) throws IOException {
        AssetFileDescriptor fd = context.getAssets().openFd(modelName);
        FileInputStream inputStream = new FileInputStream(fd.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fd.getStartOffset();
        long declaredLength = fd.getDeclaredLength();
        MappedByteBuffer retFile = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        fd.close();
        return retFile;
    }

    public List<Obj> runSegModel(Bitmap bitmap, float confThreshold, float nmsThreshold, int ogImgHeight, int ogImgWidth) {
        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
        tensorImage.load(bitmap);

        TensorImage processedTensor = imageProcessor.process(tensorImage);
        TensorBuffer output0Buffer = TensorBuffer.createFixedSize(new int[]{1, segNumAttributes, segNumAnchors}, DataType.FLOAT32);
        TensorBuffer output1Buffer = TensorBuffer.createFixedSize(new int[]{1, SEG_H, SEG_W, SEG_CHANNELS}, DataType.FLOAT32);

        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, output0Buffer.getBuffer());
        outputMap.put(1, output1Buffer.getBuffer());

        seg_interpreter.runForMultipleInputsOutputs(new Object[]{processedTensor.getBuffer()}, outputMap);

        float[] output0Flat = output0Buffer.getFloatArray();
        float[] output1Flat = output1Buffer.getFloatArray();

        float[][][][] output1 = new float[1][SEG_H][SEG_W][SEG_CHANNELS];
        int index1 = 0;
        for (int j = 0; j < SEG_H; j++) {
            for (int k = 0; k < SEG_W; k++) {
                for (int l = 0; l < SEG_CHANNELS; l++) {
                    output1[0][j][k][l] = output1Flat[index1++];
                }
            }
        }

        Object[] outputs = new Object[]{output0Flat, output1};

        return postProcessSegmentation(outputs, confThreshold, nmsThreshold, ogImgWidth, ogImgHeight);

    }


    private List<Obj> postProcessSegmentation(Object[] outputs, float confThreshold, float nmsThreshold, int ogImgWidth, int ogImgHeight) {

        float[] output0 = (float[]) outputs[0];
        float[][][][] output1 = (float[][][][]) outputs[1];

        int numClasses = segNumAttributes - SEG_CHANNELS - 4;

        float[][][] original = output1[0];
        float[][][] reshaped = new float[SEG_CHANNELS][SEG_H][SEG_W];
        float[] output1Flattened = new float[SEG_CHANNELS * SEG_H * SEG_W];

        //int index = 0;
        for (int i = 0; i < SEG_H; i++) {
            for (int j = 0; j < SEG_W; j++) {
                for (int k = 0; k < SEG_CHANNELS; k++) {
                    reshaped[k][i][j] = original[i][j][k];
                    //output1Flattened[index++] = output1[0][i][j][k];
                }
            }
        }

        int index = 0;
        for (int k = 0; k < SEG_CHANNELS; k++) {
            for (int i = 0; i < SEG_H; i++) {
                for (int j = 0; j < SEG_W; j++) {
                    output1Flattened[index++] = reshaped[k][i][j];
                }
            }
        }

        Mat output = new Mat(segNumAttributes, segNumAnchors, CvType.CV_32F);
        output.put(0, 0, output0);
        transpose(output, output); // (8400, 37)

        Mat protos = new Mat(SEG_CHANNELS, SEG_H * SEG_W, CvType.CV_32F);
        protos.put(0, 0, output1Flattened);

        List<Integer> labels = new ArrayList<>();
        List<Float> scores = new ArrayList<>();
        List<Rect> bboxes = new ArrayList<>();
        List<Mat> maskConfs = new ArrayList<>();

        float mRatio = 1.0f / (Math.min(YOLO_WIDTH / (float) ogImgWidth, YOLO_HEIGHT / (float) ogImgHeight));

        for (int i = 0; i < segNumAnchors; i++) {
            float[] row = new float[segNumAttributes];
            output.get(i, 0, row);

            float[] bboxesArr = new float[4];
            System.arraycopy(row, 0, bboxesArr, 0, 4);

            float[] scoresArr = new float[numClasses];
            System.arraycopy(row, 4, scoresArr, 0, numClasses);

            float[] maskConfsUnNorm = new float[SEG_CHANNELS];
            System.arraycopy(row, 4 + numClasses, maskConfsUnNorm, 0, SEG_CHANNELS);
            float[] maskConfsArr = new float[SEG_CHANNELS];
            for (int j = 0; j < maskConfsUnNorm.length; j++){
                maskConfsArr[j] = maskConfsUnNorm[j] * 255.0f;
            }

            float maxScore = -1;
            int maxIndex = -1;
            for (int j = 0; j < numClasses; j++) {
                if (scoresArr[j] > maxScore) {
                    maxScore = scoresArr[j];
                    maxIndex = j;
                }
            }

            if (maxScore > confThreshold) {
                float x = bboxesArr[0];
                float y = bboxesArr[1];
                float w = bboxesArr[2];
                float h = bboxesArr[3];

                float x0 = clamp((x - 0.5f * w) * ogImgWidth, 0.0f, ogImgWidth);
                float y0 = clamp((y - 0.5f * h) * ogImgHeight, 0.0f, ogImgHeight);
                float x1 = clamp((x + 0.5f * w) * ogImgWidth, 0.0f,  ogImgWidth);
                float y1 = clamp((y + 0.5f * h) * ogImgHeight, 0.0f, ogImgHeight);

                Rect bbox = new Rect((int) x0, (int) y0, (int) (x1 - x0), (int) (y1 - y0));
                Mat maskConf = new Mat(1, SEG_CHANNELS, CvType.CV_32F);
                maskConf.put(0,0,maskConfsArr);

                labels.add(maxIndex);
                scores.add(maxScore);
                bboxes.add(bbox);
                maskConfs.add(maskConf);
            }
        }

        MatOfRect2d matOfBboxes = new MatOfRect2d();
        Rect2d[] rect2dArray = new Rect2d[bboxes.size()];
        for (int i = 0; i < bboxes.size(); i++) {
            Rect rect = bboxes.get(i);
            rect2dArray[i] = new Rect2d(rect.x, rect.y, rect.width, rect.height);
        }
        matOfBboxes.fromArray(rect2dArray);

        MatOfFloat matOfScores = new MatOfFloat();
        float[] scoreArray = new float[scores.size()];
        for (int i = 0; i < scores.size(); i++) {
            scoreArray[i] = scores.get(i);
        }
        matOfScores.fromArray(scoreArray);

        MatOfInt indicesMat = new MatOfInt();

        MatOfInt matOfLabels = new MatOfInt();
        int[] labelArray = new int[labels.size()];
        for (int i = 0; i < labels.size(); i++) {
            labelArray[i] = labels.get(i);
        }
        matOfLabels.fromArray(labelArray);

        Dnn.NMSBoxesBatched(matOfBboxes, matOfScores, matOfLabels, confThreshold, nmsThreshold, indicesMat);

        List<Integer> indices = indicesMat.toList();

        Mat masks = new Mat();
        List<Obj> objs = new ArrayList<>();
        int cnt = 0;
        for (int i : indices) {
            if (cnt >= TOP_K) break;

            Rect tmp = bboxes.get(i);
            Obj obj = new Obj();
            obj.label = labels.get(i);
            obj.rect = tmp;
            obj.probability = scores.get(i);

            masks.push_back(maskConfs.get(i));
            objs.add(obj);
            cnt++;
        }

        if (!masks.empty()) {
            Mat matmulRes = new Mat();
            gemm(masks, protos, 1.0, new Mat(), 0.0, matmulRes);
            matmulRes = matmulRes.t();

            Mat maskMat = matmulRes.reshape(indices.size(), new int[]{SEG_W, SEG_H});

            List<Mat> maskChannels = new ArrayList<>();
            split(maskMat, maskChannels);

            Rect roi;

            if(ogImgHeight > ogImgWidth){
                roi = new Rect(0,0, SEG_W * ogImgWidth/ogImgHeight, SEG_H);
            }else{
                roi = new Rect(0, 0, SEG_W, SEG_H * ogImgHeight / ogImgWidth);
            }

            for (int i = 0; i < indices.size(); i++) {
                Mat dest = new Mat();
                Mat mask = new Mat();

                Mat maskChannel = maskChannels.get(i);


                multiply(maskChannel, new Scalar(-1), maskChannel);
                exp(maskChannel, dest);

                Mat one = Mat.ones(dest.size(), dest.type());
                add(dest, one, dest);
                divide(one, dest, dest);

                Mat maskTemp = new Mat(dest, roi);
                Imgproc.resize(maskTemp, mask, new Size(ogImgWidth, ogImgHeight), INTER_LINEAR);

                Mat subMatrix = mask.submat(objs.get(i).rect);
                Mat binaryMask = new Mat();
                compare(subMatrix, new Scalar(SEGMENTATION_THRESHOLD), binaryMask, Core.CMP_GT);
                objs.get(i).boxMask = binaryMask;
            }
        }

        return objs;
    }


}
