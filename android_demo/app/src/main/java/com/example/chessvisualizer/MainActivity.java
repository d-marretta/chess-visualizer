package com.example.chessvisualizer;

import static org.opencv.core.Core.addWeighted;
import static org.opencv.core.Core.perspectiveTransform;
import static java.util.Map.entry;

import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.PictureDrawable;
import android.os.Bundle;
import android.Manifest;

import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.android.OpenCVLoader;
import org.opencv.utils.Converters;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.PopupMenu;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

import com.caverock.androidsvg.SVG;
import com.caverock.androidsvg.SVGParseException;
import com.chaquo.python.PyObject;
import com.chaquo.python.Python;


public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "MainActivity";
    private static final int FRAME_INTERVAL = 15;

    private static final Map<Integer, String> CLASSES = Map.ofEntries(  entry(0,"P"),
                                                                        entry(1,"R"),
                                                                        entry(2,"N"),
                                                                        entry(3,"B"),
                                                                        entry(4,"Q"),
                                                                        entry(5,"K"),
                                                                        entry(6,"p"),
                                                                        entry(7,"r"),
                                                                        entry(8,"n"),
                                                                        entry(9,"b"),
                                                                        entry(10,"q"),
                                                                        entry(11,"k")  );

    public static class Piece{
        int row;
        int col;
        String type;
    }

    private Network net;
    private CameraBridgeViewBase cameraView;
    private ImageView processedImageView;
    private int frameCounter = 0;
    private PyObject chess_renderer;
    private String whiteOrientation;

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
        Python python = Python.getInstance();
        chess_renderer = python.getModule("renderer");

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

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat frame = inputFrame.rgba();
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);

        frameCounter++;
        if (frameCounter % FRAME_INTERVAL == 0) {

            String svgStr = getChessboardSVG(frame);
            //svgStr = "";

            if(!svgStr.isEmpty()) {
                SVG svg;
                try {
                    svg = SVG.getFromString(svgStr);
                } catch (SVGParseException e) {
                    throw new RuntimeException(e);
                }
                Drawable drawable = new PictureDrawable(svg.renderToPicture());
                runOnUiThread(() -> {
                    processedImageView.setImageDrawable(drawable);
                });
            }

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
                whiteOrientation = "top";
                return true;
            } else if (itemId == R.id.bottom) {
                Toast.makeText(this, "Bottom location selected", Toast.LENGTH_SHORT).show();
                whiteOrientation = "bottom";
                return true;
            } else if (itemId == R.id.left) {
                Toast.makeText(this, "Left location selected", Toast.LENGTH_SHORT).show();
                whiteOrientation = "left";
                return true;
            } else if (itemId == R.id.right) {
                Toast.makeText(this, "Right location selected", Toast.LENGTH_SHORT).show();
                whiteOrientation = "right";
                return true;
            }
            return false;
        });
        popupMenu.show();
    }

    private String getChessboardSVG(Mat input){
        int originalWidth = input.cols();
        int originalHeight = input.rows();

        float scale = Math.min((float) Network.YOLO_WIDTH / originalWidth, (float) Network.YOLO_HEIGHT / originalHeight);
        int newWidth = Math.round(originalWidth * scale);
        int newHeight = Math.round(originalHeight * scale);

        Mat resizedImage = new Mat();
        Imgproc.resize(input, resizedImage, new Size(newWidth, newHeight));

        int deltaWidth = Network.YOLO_WIDTH - newWidth;
        int deltaHeight = Network.YOLO_HEIGHT - newHeight;

        Mat paddedImage = new Mat();
        Core.copyMakeBorder(resizedImage, paddedImage,
                0, deltaHeight,
                0, deltaWidth,
                Core.BORDER_CONSTANT, Scalar.all(0));

        Bitmap inputBmp = Bitmap.createBitmap(paddedImage.cols(), paddedImage.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(paddedImage, inputBmp);

        List<Network.ObjSeg> resultsSeg = net.runSegModel(inputBmp, 0.25f, 0.65f, originalWidth, originalHeight);
        if (resultsSeg.isEmpty()){
            return "";
        }
        Mat mask = new Mat(input.size(), CvType.CV_8UC3, new Scalar(0, 0, 0));

        Network.ObjSeg objSeg = resultsSeg.get(0);
        Mat roi = mask.submat(objSeg.rect);
        roi.setTo(new Scalar(255,255,255), objSeg.boxMask);
        roi.release();

        List<MatOfPoint> contoursMat = new ArrayList<>();
        Mat hierarchy = new Mat();

        Imgproc.cvtColor(mask, mask, Imgproc.COLOR_RGB2GRAY);
        Imgproc.findContours(mask, contoursMat, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        List<Point> allPoints = new ArrayList<>();
        for (MatOfPoint matOfPoint : contoursMat) {
            List<Point> points = matOfPoint.toList();
            allPoints.addAll(points);
        }
        MatOfPoint2f allPoints2f = new MatOfPoint2f();
        allPoints2f.fromList(allPoints);
        MatOfPoint2f verticesMat = new MatOfPoint2f();
        Imgproc.approxPolyDP(allPoints2f, verticesMat, 0.02*Imgproc.arcLength(allPoints2f, true), true);

        List<Point> vertices = orderVertices(verticesMat.toList());
        Object[] warpResults = warpChessboard(vertices);
        Mat M = (Mat) warpResults[0];
        double squareSize = (double) warpResults[1];

        List<Network.ObjDet> resultsDet = net.runDetModel(inputBmp, CLASSES.size(),0.30f, 0.65f, originalWidth, originalHeight);

        List<Piece> piecesPositions = getPiecesPositions(resultsDet, M, squareSize);
        String fenBoard = getFenBoard(piecesPositions);
        PyObject svg = chess_renderer.callAttr("render_chessboard", fenBoard, whiteOrientation);
        return svg.toString();
    }

    private List<Point> orderVertices(List<Point> vertices) {
        List<Point> sortedVertices = new ArrayList<>();
        vertices.sort((v1, v2) -> {
            if (v1.x != v2.x) {
                return Double.compare(v2.x, v1.x);
            }
            return Double.compare(v1.y, v2.y);
        });

        List<Point> rightVertices = vertices.subList(0, 2);
        List<Point> leftVertices = vertices.subList(vertices.size()-2, vertices.size());

        Point topRight = Collections.min(rightVertices, Comparator.comparingDouble(v -> v.y));
        Point bottomRight = Collections.max(rightVertices, Comparator.comparingDouble(v -> v.y));
        Point topLeft = Collections.min(leftVertices, Comparator.comparingDouble(v -> v.y));
        Point bottomLeft = Collections.max(leftVertices, Comparator.comparingDouble(v -> v.y));

        sortedVertices.add(topLeft);
        sortedVertices.add(topRight);
        sortedVertices.add(bottomRight);
        sortedVertices.add(bottomLeft);

        return sortedVertices;
    }

    private Object[] warpChessboard(List<Point> vertices){
        Point topLeft = vertices.get(0);
        Point topRight = vertices.get(1);

        double sideChessboard = Math.sqrt(Math.pow( (topRight.x - topLeft.x), 2) + Math.pow( (topRight.y - topLeft.y), 2));

        List<Point> dest = new ArrayList<>();
        dest.add(new Point(0,0));
        dest.add(new Point(sideChessboard,0));
        dest.add(new Point(sideChessboard,sideChessboard));
        dest.add(new Point(0,sideChessboard));

        Mat srcMat = Converters.vector_Point2f_to_Mat(vertices);
        Mat dstMat = Converters.vector_Point2f_to_Mat(dest);

        Mat M = Imgproc.getPerspectiveTransform(srcMat, dstMat);
        double squareSize = sideChessboard / 8;

        return new Object[]{M, squareSize};
    }

    private List<Piece> getPiecesPositions(List<Network.ObjDet> objs, Mat M, double square_size){
        List<Point> ogPoints = new ArrayList<>();
        List<String> pieceClasses = new ArrayList<>();
        for (Network.ObjDet obj : objs){
            Rect r = obj.rect;
            Point lowCenter = new Point(r.x + r.width /2f, r.y + r.height/2f + r.height/2.4);
            ogPoints.add(lowCenter);

            String pieceClass = CLASSES.get(obj.label);
            pieceClasses.add(pieceClass);
        }

        Mat transformedPointsMat = new Mat();
        perspectiveTransform(Converters.vector_Point2f_to_Mat(ogPoints), transformedPointsMat, M);
        List<Point> transformedPoints = new ArrayList<>();
        Converters.Mat_to_vector_Point2f(transformedPointsMat, transformedPoints);

        List<Piece> piecesPositions = new ArrayList<>();
        for (int i = 0; i < transformedPoints.size(); i++){
            double transformedX = transformedPoints.get(i).x;
            double transformedY = transformedPoints.get(i).y;

            int square_col = (int) (transformedX / square_size);
            int square_row = (int) (transformedY / square_size) + 1;
            Piece piece = new Piece();
            piece.col = square_col;
            piece.row = 8-square_row;
            piece.type = pieceClasses.get(i);
            piecesPositions.add(piece);
        }
        return piecesPositions;

    }

    private String getFenBoard(List<Piece> positions){
        String[][] board = new String[8][8];
        for (int i = 0; i < 8; i++){
            for (int j = 0; j < 8; j++){
                board[i][j] = "";
            }
        }

        for (Piece p : positions){
            try {
                board[p.row][p.col] = p.type;
            } catch(IndexOutOfBoundsException e){}
        }

        StringBuilder fenRows = new StringBuilder();

        for (int i = 7; i >= 0; i--){
            StringBuilder fenRow = new StringBuilder();
            int emptyCount = 0;

            for (int j = 0; j < 8; j++){
                String cell = board[i][j];
                if (cell.isEmpty()) {
                    emptyCount++;
                }else{
                    if (emptyCount > 0){
                        fenRow.append(emptyCount);
                        emptyCount = 0;
                    }
                    fenRow.append(cell);
                }
            }
            if (emptyCount > 0){
                fenRow.append(emptyCount);
            }
            fenRows.append(fenRow.toString()).append('/');
        }
        return fenRows.substring(0, fenRows.length()-1);

    }
}