#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(void)
{
	const int img_num = 1;		// image number
	const int row_num = 9;			// number of corner on row
	const int colum_num = 6;		// numner of corner on colum

	Size board_size = Size(row_num, colum_num);

	vector<Point2f> corners;		// save corners in one image
	vector<vector<Point2f>> corners_Seq;		// save corners of all images

	vector<Mat> img_Seq;			// save images

	// find corners
	for (int i = 0; i < img_num; i++)
	{
		string img_name = "1" + i;
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
			
			string img_corner_name = "1" + i;
			img_corner_name = "pic/" + img_corner_name;
			img_corner_name += "_corner.jpg";

			imwrite(img_corner_name, img_corner);
		}

		else
		{
			cout << "No corner is found in " << img_name << endl;
			getchar();
			exit(1);
		}
	}

	return 0;
}