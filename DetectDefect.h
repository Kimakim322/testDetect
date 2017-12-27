#pragma once
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include <vector>
#include <algorithm>
#include <iostream>
#include <memory>
#include <ctime>

using namespace std;
using namespace cv;

struct ReturnValue
{
	vector<vector<Point> > retValue;
	int lineLength;
};


class DetectDefect
{
public:
	DetectDefect();
	~DetectDefect();
	//void Thread1(void*);
	//void Thread2(void*);
	//void Thread3(void*);
	//void Thread4(void*);
	void Kontrast(Mat& src, int step);
	void Rezkost(Mat& src, float step);
	void DrawCntrs(Mat& src, int a, int b, int cl);
	Mat DrawLargeCntrs(Mat& src, int a, int b, int cl);
	Mat DrawDefects(Mat& canny2);
	double angle(Point pt1, Point pt2, Point pt0);
	void findSquares(Mat& image, vector<vector<Point> >& squares);
	void drawSquaresFromSmall(Mat& image, vector<vector<Point> >& squares, Mat& bigImage);
	ReturnValue GetDefect(std::string strName);
};



