package com.example.dana.myapplication;

import android.content.Context;
import android.content.ContextWrapper;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.FaceDetector;
import android.os.Bundle;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.Date;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private static final String TAG = "OCVSample::Activity";
    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
    public static final int JAVA_DETECTOR = 0;
    public static final int NATIVE_DETECTOR = 1;

    private MenuItem mItemFace50;
    private MenuItem mItemFace40;
    private MenuItem mItemFace30;
    private MenuItem mItemFace20;
    private MenuItem mItemType;

    private Mat mRgba;
    private Mat mGray;

    private int mDetectorType = JAVA_DETECTOR;
    private String[] mDetectorName;

    private float mRelativeFaceSize = 0.2f;
    private int mAbsoluteFaceSize = 0;

    private JavaCameraView mOpenCvCameraView;
    private Button mCapturePicture;
    private Button mTemplatePicture;
    private Mat imageMat;

    public MainActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
        mDetectorName[NATIVE_DETECTOR] = "Native (tracking)";
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i("OpenCV", "OpenCV loaded successfully");
                    //System.loadLibrary("detection_based_tracker");
                    //test();
                    test2();
                    imageMat = new Mat();
                    mOpenCvCameraView.setCameraIndex(1);
                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    static {
        if (!OpenCVLoader.initDebug()) {
            // Handle initialization error
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        super.onCreate(savedInstanceState);

        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        //OpenCVLoader.initDebug();

        mOpenCvCameraView = (JavaCameraView) findViewById(R.id.view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        mCapturePicture = (Button) findViewById(R.id.captureButton);
        mTemplatePicture = (Button) findViewById(R.id.templateButton);
        mCapturePicture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                takeAPicture(false);
            }
        });
        mTemplatePicture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                takeAPicture(true);
            }
        });
    }

    String mSourceFile;

    private void takeAPicture(boolean isASource) {
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
        String currentDateandTime = sdf.format(new Date());
        //String postfix = "sample_picture_" + currentDateandTime + ".jpg";
        String postfix = "sample_picture.jpg";
        String fileName = Environment.getExternalStorageDirectory().getPath() + postfix;
        mOpenCvCameraView.takePicture(postfix, this);
        if (isASource) {
            mSourceFile = postfix;
            recognizeFace();
        } else { // if capture comparable picture
            if (mSourceFile != null) {
                // here to compare both
                recognizeFace();
            }
        }
    }

    private void recognizeFace() {
        //Bitmap picture = getPictureFromStorage(mSourceFile);
        ContextWrapper cw = new ContextWrapper(this);
        File directory = cw.getDir("imageDir", Context.MODE_PRIVATE);
        Bitmap picture = loadImageFromStorage(directory.getPath(), mSourceFile);
        Mat mat = new Mat(); // matrix to store photo
        Utils.bitmapToMat(picture, mat);
        //Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGB2GRAY, 4); // convert matrix from rgb to gray style
        // FaceDetector faceDetector = new FaceDetector();
        test2();
        Imgproc.cvtColor(mat, grayscaleImage, Imgproc.COLOR_RGBA2GRAY);
        MatOfRect faces = new MatOfRect();
        // Use the classifier to detect faces
        //cascade.detectMultiScale(grayImage, objects, 1.1, 3, CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_CANNY_PRUNING,cvSize(0,0), cvSize(100,100));

        if (cascadeClassifier != null) {
            cascadeClassifier.detectMultiScale(grayscaleImage, faces, 1.1, 1, 0,
                    new Size(0, 0), new Size(absoluteFaceSize, absoluteFaceSize));
        }
        // If there are any faces found, draw a rectangle around it
        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++)
            Imgproc.rectangle(mat, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0, 255), 3);
        //Core.rectangle(aInputFrame, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0, 255), 3);
    }

    private File mCascadeFile;
    private CascadeClassifier mJavaDetector;
    private CascadeClassifier cascadeClassifier;
    private DetectionBasedTracker mNativeDetector;

    private void test() {
        try {
            InputStream is = this.getResources().openRawResource(R.raw.lbpcascade_frontalface);
            File cascadeDir = this.getDir("cascade", Context.MODE_PRIVATE);
            mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;

            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            if (mJavaDetector.empty()) {
                Log.e("OpenCV", "Failed to load cascade classifier");
                mJavaDetector = null;
            } else
                Log.i("OpenCV", "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

            mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0);

            cascadeDir.delete();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    private void test2() {
        try {
            // Copy the resource into a temp file so OpenCV can load it
            InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            // Load the cascade classifier
            cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            if (!cascadeClassifier.empty()) {
                Log.i("OpenCV","cascade classifier successfully created from: " + mCascadeFile.getAbsolutePath());
            }
            cascadeDir.delete();
        } catch (Exception e) {
            Log.e("OpenCVActivity", "Error loading cascade", e);
        }
    }

    private Bitmap loadImageFromStorage(String path, String fileName) {
        Bitmap b = null;
        try {
            File f=new File(path, fileName);
            b = BitmapFactory.decodeStream(new FileInputStream(f));
        }
        catch (FileNotFoundException e)
        {
            e.printStackTrace();
        }
        return b;
    }

    private Bitmap getPictureFromStorage(String mSourceFile) {
        Bitmap picture = null;
        File file = new File(getFilesDir(), mSourceFile); // Pass getFilesDir() and "MyFile" to read file
        try {
            FileInputStream fileInputStream = new FileInputStream(file);

        String photoPath = Environment.getExternalStorageDirectory() + mSourceFile;

        BitmapFactory.Options opts = new BitmapFactory.Options();
        opts.inPreferredConfig = Bitmap.Config.ARGB_8888;
        picture = BitmapFactory.decodeStream(fileInputStream);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        return picture;
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d("OpenCV", "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    private Mat grayscaleImage;
    private int absoluteFaceSize;

    @Override
    public void onCameraViewStarted(int width, int height) {
        /*mGray = new Mat();
        mRgba = new Mat();*/
        grayscaleImage = new Mat(height, width, CvType.CV_8UC4);
        // The faces will be a 70% of the height of the screen
        absoluteFaceSize = (int) (height * 0.5);
    }

    @Override
    public void onCameraViewStopped() {
       /* mGray.release();
        mRgba.release();*/
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
       /* mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
            mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
        }

        MatOfRect faces = new MatOfRect();

        if (mDetectorType == JAVA_DETECTOR) {
            if (mJavaDetector != null)
                mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2,
                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        }
        else if (mDetectorType == NATIVE_DETECTOR) {
            if (mNativeDetector != null)
                mNativeDetector.detect(mGray, faces);
        }
        else {
            Log.e(TAG, "Detection method is not selected!");
        }

        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++)
            Imgproc.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);

        return mRgba;*/


        // Create a grayscale image
/*        Mat aInputFrame = inputFrame.rgba();
        Imgproc.cvtColor(aInputFrame, grayscaleImage, Imgproc.COLOR_RGBA2GRAY);
        MatOfRect faces = new MatOfRect();
        // Use the classifier to detect faces
        //cascade.detectMultiScale(grayImage, objects, 1.1, 3, CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_CANNY_PRUNING,cvSize(0,0), cvSize(100,100));

        if (cascadeClassifier != null) {
            cascadeClassifier.detectMultiScale(grayscaleImage, faces, 1.1, 1, 0,
                    new Size(0, 0), new Size(absoluteFaceSize, absoluteFaceSize));
        }
        // If there are any faces found, draw a rectangle around it
        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++)
            Imgproc.rectangle(aInputFrame, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0, 255), 3);
            //Core.rectangle(aInputFrame, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0, 255), 3);
        return aInputFrame;*/


        imageMat = inputFrame.rgba();
        Mat mRgbaT = imageMat.t();
        Core.flip(imageMat.t(), mRgbaT, 1);
        Imgproc.resize(mRgbaT, mRgbaT, imageMat.size());
        return mRgbaT;
    }
}
