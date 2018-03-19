#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(void)
{
	ofstream fout("calibration_result.txt");		// ����궨���

	const int img_num = 14;			// image number
	const int row_num = 9;			// number of corner on row
	const int colum_num = 6;		// numner of corner on colum

	Size board_size = Size(row_num, colum_num);

	vector<Point2f> corners;					// save corners in one image
	vector<vector<Point2f>> corners_Seq;		// save corners of all images

	vector<Mat> img_Seq;						// save images

	// ��ȡͼ���еĽǵ㣬�����������ؾ�ȷ��
	for (int i = 0; i < img_num; i++)
	{
		string img_name;
		stringstream StrStm;
		StrStm << i + 1;
		StrStm >> img_name;
		img_name = "pic/" + img_name;
		img_name += ".jpg";

		Mat img = imread(img_name);
		Mat img_gray;

		cvtColor(img, img_gray, CV_RGB2GRAY);

		bool patternfound = findChessboardCorners(img_gray, board_size, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);

		if (patternfound)
		{
			// �����ؾ�ȷ��
			cornerSubPix(img_gray, corners, Size(11,11), Size(-1,-1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			Mat img_corner = img.clone();
			drawChessboardCorners(img_corner, board_size, Mat(corners), patternfound);
			
			string img_corner_name;
			stringstream StrStm;
			StrStm << i + 1;
			StrStm >> img_corner_name;
			img_corner_name = "pic/corner/" + img_corner_name;
			img_corner_name += "_corner.jpg";
			
			imwrite(img_corner_name, img_corner);
			
			corners_Seq.push_back(corners);
			img_Seq.push_back(img);
		}

		else
		{
			cout << "No corner is found in " << img_name << endl;
			getchar();
			exit(1);
		}
	}

	// ����궨
	cout << "��ʼ�궨��" << endl;

	Size square_size = Size(20, 20);			// �궨��С����ĳߴ�
	vector<vector<Point3f>> objects_Points;		// ����궨���ϵ���ά����

	// ��ʼ���궨����ÿ���ǵ����ά����
	for (int t = 0; t < img_Seq.size(); t++)
	{
		vector<Point3f> tempPointSet;
		for (int i = 0; i < board_size.height; i++)
		{
			for (int j = 0; j < board_size.width; j++)
			{
				// �궨���������������ϵ��z=0��ƽ����
				Point3f tempPoint;
				tempPoint.x = i*square_size.width;
				tempPoint.y = j*square_size.height;
				tempPoint.z = 0;
				tempPointSet.push_back(tempPoint);
			}
		}
		objects_Points.push_back(tempPointSet);
	}

	// ��ʼ�궨

	Size img_size = img_Seq[0].size();
	Matx33d intrinsic_matrix;				// ����ڲξ���
	Vec4d distortion_coeffs;				// �����4������ϵ����K1��K2��K3��K4
	vector<Vec3d> rotation_vectors;			// ÿ��ͼ�����ת����
	vector<Vec3d> translation_vectors;		// ÿ��ͼ���ƽ������
	int flags = 0;

	flags |= fisheye::CALIB_RECOMPUTE_EXTRINSIC;
	flags |= fisheye::CALIB_CHECK_COND;
	flags |= fisheye::CALIB_FIX_SKEW;

	fisheye::calibrate(objects_Points, corners_Seq, img_size, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, flags, TermCriteria(3, 20, 1e-6));

	cout << "�궨��ɣ�" << endl;

	// ���۱궨���
	cout << "��ʼ���۱궨�����" << endl;
	double total_err = 0.0;					// ����ͼ���ƽ������ܺ�
	double err = 0.0;						// ÿ��ͼ���ƽ�����
	vector<Point2f> img_points2;			// �����ؼ���õ���ͶӰ��

	cout << "ÿ��ͼ��ı궨��" << endl;

	for (int i = 0; i < img_Seq.size(); i++)
	{
		vector<Point3f> tempPointSet = objects_Points[i];
		// ͨ������õ�������������Կռ����ά�������ͶӰ���㣬�õ��µ�ͶӰ��
		fisheye::projectPoints(tempPointSet, img_points2, rotation_vectors[i], translation_vectors[i], intrinsic_matrix, distortion_coeffs);
		// �����¾�ͶӰ��֮������
		vector<Point2f> tempImagePoint = corners_Seq[i];

		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat image_points2Mat = Mat(1, img_points2.size(), CV_32FC2);

		for (size_t i = 0; i != tempImagePoint.size(); i++)
		{
			image_points2Mat.at<Vec2f>(0, i) = Vec2f(img_points2[i].x, img_points2[i].y);
			tempImagePointMat.at<Vec2f>(0, i) = Vec2f(tempImagePoint[i].x, tempImagePoint[i].y);
		}

		err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
		total_err += err /= board_size.height * board_size.width;
		
		cout << "��" << i + 1 << "��ͼ���ƽ����" << err << "����" << endl;

	}
	cout << "����ƽ����" << total_err / img_Seq.size() << "����" << endl;

	// ����궨���
	cout << "����궨���" << endl;
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0));				// ����ÿ��ͼ�����ת����
	fout << "����ڲξ���" << endl;
	fout << intrinsic_matrix << endl;
	fout << "����ϵ��" << endl;
	fout << distortion_coeffs << endl;

	cout << "�������" << endl;

	// ����У��ͼ��

	Mat mapx = Mat(img_size, CV_32FC1);
	Mat mapy = Mat(img_size, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32F);

	cout << "����У��ͼ��" << endl; 
	for (int i = 0; i < img_Seq.size(); i++)
	{
		fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R,
			getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coeffs, img_size, 1, img_size, 0), img_size, CV_32FC1, mapx, mapy);
		
		Mat t = img_Seq[i].clone();
		remap(img_Seq[i], t, mapx, mapy, INTER_LINEAR);
		
		string undistoration_img_name;
		stringstream StrStm;
		StrStm << i + 1;
		StrStm >> undistoration_img_name;
		undistoration_img_name = "pic/undistoration/" + undistoration_img_name;
		undistoration_img_name += "_d.jpg";
		imwrite(undistoration_img_name, t);

	}
		

	return 0;
}