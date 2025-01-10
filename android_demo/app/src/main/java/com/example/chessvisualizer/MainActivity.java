package com.example.chessvisualizer;

import static org.opencv.core.Core.addWeighted;

import android.content.Context;
import android.content.ContextWrapper;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.Manifest;

import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.android.OpenCVLoader;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.PopupMenu;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;


public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "MainActivity";

    private Network net;
    private CameraBridgeViewBase cameraView;
    private ImageView processedImageView;
    private Mat grayMat;
    private int frameCounter = 0;
    private static final int FRAME_INTERVAL = 10;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 1);
        } else {
            initializeNetwork();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 1) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                initializeApp();
            } else {
                Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    private void initializeNetwork() {
        try {
            net = new Network(this, "yolo11n-seg-best_float32.tflite", "yolo11n-best_float32.tflite", new Network.InitializationCallback() {
                @Override
                public void onInitializationComplete() {
                    initializeApp();
                }

                @Override
                public void onInitializationFailed(Exception e) {
                    Log.e(TAG, "Network initialization failed: " + e.getMessage());
                    runOnUiThread(() -> {
                        Toast.makeText(MainActivity.this, "Network initialization failed", Toast.LENGTH_SHORT).show();
                        finish();
                    });
                }
            });
        } catch (IOException e) {
            Log.e(TAG, "Network initialization failed: " + e.getMessage());
            Toast.makeText(this, "Network initialization failed", Toast.LENGTH_SHORT).show();
            finish();
        }
    }

    private void initializeApp() {
        if (OpenCVLoader.initLocal()) {
            Log.i(TAG, "OpenCV loaded successfully");
        } else {
            Log.e(TAG, "OpenCV initialization failed!");
            Toast.makeText(this, "OpenCV initialization failed!", Toast.LENGTH_LONG).show();
            return;
        }

        setContentView(R.layout.activity_main);

        cameraView = findViewById(R.id.camera_view);
        processedImageView = findViewById(R.id.processed_image_view);
        cameraView.setCameraPermissionGranted();

        cameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        cameraView.setCvCameraViewListener(this);
        cameraView.enableView();

        Button optionsButton = findViewById(R.id.options_button);
        optionsButton.setOnClickListener(this::showPopupMenu);

    }

    @Override
    public void onResume() {
        super.onResume();
        if (cameraView != null)
            cameraView.enableView();
    }


    @Override
    public void onPause() {
        super.onPause();
        if (cameraView != null) {
            cameraView.disableView();
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (cameraView != null) {
            cameraView.disableView();
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        grayMat = new Mat(height, width, CvType.CV_8UC1);
    }

    @Override
    public void onCameraViewStopped() {
        if (grayMat != null) grayMat.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat frame = inputFrame.rgba();
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);

        frameCounter++;
        if (frameCounter % FRAME_INTERVAL == 0) {
            //Imgproc.cvtColor(frame, grayMat, Imgproc.COLOR_RGB2GRAY);

            runOnUiThread(() -> {

                AssetManager assetManager = getAssets();

                InputStream istr;
                Bitmap bitmap;
                try {
                    istr = assetManager.open("G000_IMG000.jpg");
                    bitmap = BitmapFactory.decodeStream(istr);
                    istr.close();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                Mat imageTemp = new Mat();
                Utils.bitmapToMat(bitmap, imageTemp);
                Log.v("ImageView size", processedImageView.getWidth() + " " + processedImageView.getHeight());
                Imgproc.resize(imageTemp, imageTemp, new Size(processedImageView.getHeight(), processedImageView.getWidth()));
                Bitmap input = Bitmap.createBitmap(imageTemp.cols(), imageTemp.height(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(imageTemp, input);
                List<Network.Obj> results = net.runSegModel(input, 0.25f, 0.65f, input.getHeight(), input.getWidth());

                Mat image = new Mat();
                Utils.bitmapToMat(input, image);
                Mat mask = image.clone();

                for (Network.Obj obj : results) {
                    Mat roi = mask.submat(obj.rect);
                    Log.v("rect", obj.rect.x + " " + obj.rect.y + " " + obj.rect.width + " " + obj.rect.height);
                    roi.setTo(new Scalar(0,0,255), obj.boxMask);
                    roi.release();

                }
                addWeighted(image, 0.5, mask, 0.8, 1.0, image);
                for (Network.Obj obj : results) {

                    Imgproc.rectangle(image, obj.rect, new Scalar(0,0,255));
                }
                Bitmap bmp = Bitmap.createBitmap(image.cols(), image.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(image, bmp);



                processedImageView.setImageBitmap(bmp);
            });
        }

        return frame;
    }

    private void showPopupMenu(View v) {
        PopupMenu popupMenu = new PopupMenu(this, v);
        popupMenu.getMenuInflater().inflate(R.menu.options_menu, popupMenu.getMenu());
        popupMenu.setOnMenuItemClickListener(item -> {
            int itemId = item.getItemId();
            if (itemId == R.id.top) {
                Toast.makeText(this, "Top location selected", Toast.LENGTH_SHORT).show();
                return true;
            } else if (itemId == R.id.bottom) {
                Toast.makeText(this, "Bottom location selected", Toast.LENGTH_SHORT).show();
                return true;
            } else if (itemId == R.id.left) {
                Toast.makeText(this, "Left location selected", Toast.LENGTH_SHORT).show();
                return true;
            } else if (itemId == R.id.right) {
                Toast.makeText(this, "Right location selected", Toast.LENGTH_SHORT).show();
                return true;
            }
            return false;
        });
        popupMenu.show();
    }
}