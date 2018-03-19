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
	ofstream fout("calibration_result.txt");		// 保存标定结果

	const int img_num = 14;			// image number
	const int row_num = 9;			// number of corner on row
	const int colum_num = 6;		// numner of corner on colum

	Size board_size = Size(row_num, colum_num);

	vector<Point2f> corners;					// save corners in one image
	vector<vector<Point2f>> corners_Seq;		// save corners of all images

	vector<Mat> img_Seq;						// save images

	// 提取图像中的角点，并进行亚像素精确化
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
			// 亚像素精确化
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

	// 相机标定
	cout << "开始标定！" << endl;

	Size square_size = Size(20, 20);			// 标定板小方格的尺寸
	vector<vector<Point3f>> objects_Points;		// 保存标定板上的三维坐标

	// 初始化标定板上每个角点的三维坐标
	for (int t = 0; t < img_Seq.size(); t++)
	{
		vector<Point3f> tempPointSet;
		for (int i = 0; i < board_size.height; i++)
		{
			for (int j = 0; j < board_size.width; j++)
			{
				// 标定板放置在世界坐标系中z=0的平面上
				Point3f tempPoint;
				tempPoint.x = i*square_size.width;
				tempPoint.y = j*square_size.height;
				tempPoint.z = 0;
				tempPointSet.push_back(tempPoint);
			}
		}
		objects_Points.push_back(tempPointSet);
	}

	// 开始标定

	Size img_size = img_Seq[0].size();
	Matx33d intrinsic_matrix;				// 相机内参矩阵
	Vec4d distortion_coeffs;				// 相机的4个畸变系数，K1，K2，K3，K4
	vector<Vec3d> rotation_vectors;			// 每幅图像的旋转向量
	vector<Vec3d> translation_vectors;		// 每幅图像的平移向量
	int flags = 0;

	flags |= fisheye::CALIB_RECOMPUTE_EXTRINSIC;
	flags |= fisheye::CALIB_CHECK_COND;
	flags |= fisheye::CALIB_FIX_SKEW;

	fisheye::calibrate(objects_Points, corners_Seq, img_size, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, flags, TermCriteria(3, 20, 1e-6));

	cout << "标定完成！" << endl;

	// 评价标定结果
	cout << "开始评价标定结果！" << endl;
	double total_err = 0.0;					// 所有图像的平均误差总和
	double err = 0.0;						// 每幅图像的平均误差
	vector<Point2f> img_points2;			// 保存重计算得到的投影点

	cout << "每幅图像的标定误差：" << endl;

	for (int i = 0; i < img_Seq.size(); i++)
	{
		vector<Point3f> tempPointSet = objects_Points[i];
		// 通过计算得到的相机参数，对空间的三维点进行重投影计算，得到新的投影点
		fisheye::projectPoints(tempPointSet, img_points2, rotation_vectors[i], translation_vectors[i], intrinsic_matrix, distortion_coeffs);
		// 计算新旧投影点之间的误差
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
		
		cout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;

	}
	cout << "总体平均误差：" << total_err / img_Seq.size() << "像素" << endl;

	// 保存标定结果
	cout << "保存标定结果" << endl;
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0));				// 保存每幅图像的旋转矩阵
	fout << "相机内参矩阵" << endl;
	fout << intrinsic_matrix << endl;
	fout << "畸变系数" << endl;
	fout << distortion_coeffs << endl;

	cout << "保存结束" << endl;

	// 保存校正图像

	Mat mapx = Mat(img_size, CV_32FC1);
	Mat mapy = Mat(img_size, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32F);

	cout << "保存校正图像" << endl; 
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