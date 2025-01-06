package com.example.chessvisualizer;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.util.Log;

import com.google.android.gms.tasks.Task;
import com.google.android.gms.tflite.client.TfLiteInitializationOptions;
import com.google.android.gms.tflite.java.TfLite;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.InterpreterApi;
import org.tensorflow.lite.gpu.GpuDelegateFactory;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;



public class Network {

    private static final float CONFIDENCE_THRESHOLD = 0.5f;
    private static final int YOLO_HEIGHT = 640;
    private static final int YOLO_WIDTH = 640;

    private InterpreterApi seg_interpreter;
    private int segNumChannel;
    private int segNumElements;
    private InterpreterApi det_interpreter;
    private ImageProcessor imageProcessor;

    public interface InitializationCallback {
        void onInitializationComplete();
        void onInitializationFailed(Exception e);
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

                    int[] outputShape = seg_interpreter.getOutputTensor(0).shape();
                    Log.d("boh", Arrays.toString(outputShape));

                    if (outputShape != null) {
                        segNumElements = outputShape[1];
                        segNumChannel = outputShape[2];
                    }
                    callback.onInitializationComplete();
                })
                .addOnFailureListener(e -> {
                    Log.e("Interpreter", String.format("Cannot initialize interpreter: %s", e.getMessage()));
                    callback.onInitializationFailed(e);
                });


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

    public float[] runSegModel(Bitmap bitmap){
        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
        tensorImage.load(bitmap);
        TensorImage processedTensor = imageProcessor.process(tensorImage);

        TensorBuffer outputBuffer = TensorBuffer.createFixedSize(new int[]{1, segNumChannel, segNumElements}, DataType.FLOAT32);
        seg_interpreter.run(processedTensor.getBuffer(), outputBuffer.getBuffer().rewind());
        Log.d("boh", Arrays.toString(outputBuffer.getFloatArray()));
        return outputBuffer.getFloatArray();
    }

    /*public List<BoundingBox> bestBox(float[] array) {
        List<BoundingBox> boundingBoxes = new ArrayList<>();
        for (int r = 0; r < numElements; r++) {
            float cnf = array[r * numChannel + 4];
            if (cnf > CONFIDENCE_THRESHOLD) {
                float x1 = array[r * numChannel];
                float y1 = array[r * numChannel + 1];
                float x2 = array[r * numChannel + 2];
                float y2 = array[r * numChannel + 3];
                int cls = (int) array[r * numChannel + 5];
                String clsName = labels.get(cls);
                boundingBoxes.add(new BoundingBox(x1, y1, x2, y2, cnf, cls, clsName));
            }
        }
        return boundingBoxes;
    }*/


    




}
