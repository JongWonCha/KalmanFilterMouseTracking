// Kalman Filter for mouse tracking
// OpenCV 4.6.0
// github@JongWonCha

#include <opencv2/highgui/highgui_c.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <tchar.h>
#include <iostream>
#include <queue>
#include <chrono>

using namespace Eigen;

CvPoint pt_Measurement;


class kalmanFilter {
public:
	Matrix<double, 6, 1> xHat;			//추정값       
	Matrix<double, 6, 6> covariance;    //오차공분산 매트릭스
	Matrix<double, 6, 6> A;				//상태전이 매트릭스
	Matrix<double, 3, 6> H;             //측정 행렬
	Matrix<double, 6, 6> processNoise;  //Q값 - 모델 오차
	Matrix<double, 3, 3> measureNoise;  //R값 - 측정 오차

	kalmanFilter(double proNoise, double meaNoise, double initX, double initY) {
		A = Matrix<double, 6, 6>::Identity();
		H = Matrix<double, 3, 6>::Zero();
		H(0, 0) = 1;
		H(1, 1) = 1;
		H(2, 2) = 1;
		xHat = Matrix<double, 6, 1>::Zero();
		covariance = Matrix<double, 6, 6>::Identity();
		processNoise = Matrix<double, 6, 6>::Identity() * proNoise;
		measureNoise = Matrix<double, 3, 3>::Identity() * meaNoise;
		xHat(0, 0) = initX;
		xHat(1, 0) = initY;
	}

	static kalmanFilter& getInstance(double proNoise, double meaNoise, double initX, double initY) {
		static kalmanFilter instance(proNoise, meaNoise, initX, initY);
		return instance;
	}

	void predict(double dt) {
		A(0, 3) = A(1, 4) = A(2, 5) = dt;
		xHat = A * xHat;
		covariance = A * covariance * A.transpose() + processNoise;
	}

	void update(const Vector3d& z) {
		Vector3d y = z - H * xHat;
		Matrix3d temp = H * covariance * H.transpose() * measureNoise;
		Matrix<double, 6, 3> kalmanGain = covariance * H.transpose() * temp.inverse();

		xHat = xHat + kalmanGain * y;
		covariance = (Matrix<double, 6, 6>::Identity() - kalmanGain * H) * covariance;
	}
	
	kalmanFilter(const kalmanFilter&) = delete;
	kalmanFilter& operator=(const kalmanFilter&) = delete;	//싱글톤 보장
};

void on_mouse(int event, int x, int y, int flags, void*) {
	pt_Measurement = cvPoint(x, y);
}

int _tmain(int argc, _TCHAR* argv[]) {	
	// Measurements are current position of the mouse [X;Y]
	CvMat* measurement = cvCreateMat(2, 1, CV_32FC1);
	auto startTime = std::chrono::system_clock::now();
	auto endTime = std::chrono::system_clock::now();
	std::chrono::system_clock::time_point start;
 
	// tracking Points vector
	std::vector <CvPoint> vt_Measurement;
	std::vector <CvPoint> vt_Prediction;

	// strings
	CvFont font1;
	cvInitFont(&font1, CV_FONT_HERSHEY_DUPLEX, 0.4, 0.4, 0, 1, CV_AA);
	char str0[20] = "Kalman Filter  ";
	char str1[20] = "Mouse Position";
	char str2[22] = "JWC KF Mouse Tracking";
	
	// etc.
	int k;
	int FrameCount = 0;
	int max_trace = 1000;
	int measuremnt_frame = 4;	// Measurement will be available at every "measuremnt_frame" frame
	int IsMeasurementExist = 0;

	// Image to show and mouse input
	IplImage* img_src = cvCreateImage(cvSize(720, 480), IPL_DEPTH_8U, 3);
	cvNamedWindow("Kalman Filter", 1);	
	cvSetMouseCallback("Kalman Filter", on_mouse, 0);

	kalmanFilter& kf = kf.getInstance(1e-4, 15, (double)img_src->width / 2, (double)img_src->height / 2);	//모델오차, 측정오차, x시작점, y시작점

	// Loop
	for(;;) {
		endTime = std::chrono::system_clock::now();
		//std::cout << kf.xHat[0] << '\t' << kf.xHat[1] << std::endl;
		measurement->data.fl[0] = (float)pt_Measurement.x;
		measurement->data.fl[1] = (float)pt_Measurement.y;

		endTime = std::chrono::system_clock::now();
		vt_Measurement.push_back(pt_Measurement);
		
		kf.predict(1);
		kf.update(Vector3d((double)pt_Measurement.x, (double)pt_Measurement.y, 0));
		CvPoint a;
		a.x = (int)kf.xHat[0];
		a.y = (int)kf.xHat[1];
		//std::cout << a.x << '\t' << a.y << std::endl;
		//std::cout << vt_Measurement.size() << std::endl;
		vt_Prediction.push_back(a);
		startTime = std::chrono::system_clock::now();

		// Clear image
		cvZero(img_src);
		
		for((int)vt_Measurement.size()>max_trace/measuremnt_frame?k=(int)vt_Measurement.size()-max_trace/measuremnt_frame:
			k=0;k<(int)vt_Measurement.size()-1;k++) {
			cvDrawCircle(img_src, vt_Measurement[k],  1, cvScalar(0,255,0), -1, 8, 0);
			cvLine(img_src, vt_Prediction[k + 1], vt_Prediction[k], cvScalar(0, 255, 255), 1, 8, 0);
		}

		// Drawing legend
		cvPutText(img_src, str0, cvPoint(img_src->width-120, 30), &font1, cvScalar(255,255,255));
		cvPutText(img_src, str1, cvPoint(img_src->width-130, 45), &font1, cvScalar(255,255,255));
		cvPutText(img_src, str2, cvPoint(img_src->width - 160, img_src->height - 10), &font1, cvScalar(255, 255, 255));
		cvLine(img_src, cvPoint(img_src->width-16, 26), cvPoint(img_src->width-24, 26), cvScalar(0,255,255), 2, 8, 0);
		cvDrawCircle(img_src, cvPoint(img_src->width-20, 41), 4, cvScalar(0,255,0), -1, 8, 0);

		// Show image
		cvShowImage("Kalman Filter", img_src);

		// Key input
		if(cvWaitKey(1)>0)	
			break;

		FrameCount++;
		
	}

	// Releases
	cvDestroyAllWindows();
	cvReleaseImage(&img_src);

	return 0;
}