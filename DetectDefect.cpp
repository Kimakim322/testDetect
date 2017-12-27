#include "DetectDefect.h"
#include <vector>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>
using namespace cv;
using namespace std;

struct timeval tv1,tv2,dtv;
struct timezone tz;
void time_start() { gettimeofday(&tv1, &tz); }
long time_stop()
{ gettimeofday(&tv2, &tz);
  dtv.tv_sec= tv2.tv_sec -tv1.tv_sec;
  dtv.tv_usec=tv2.tv_usec-tv1.tv_usec;
  if(dtv.tv_usec<0) { dtv.tv_sec--; dtv.tv_usec+=1000000; }
  return dtv.tv_sec*1000+dtv.tv_usec/1000;
}

int thresh = 50, N = 11;
const char* wndname = "Square Detection Demo";
int coefDel = 20;

double coefCosine = 0.3;
double coefBigImg = 1.5;
double coefAfterTresh = 3.0;
double firstCoefBigImg = 1.0;
int wallXmin = 100;
int wallYmin = 100;
int wallXmax = 400;
int wallYmax = 500;
int ramX = 50;
int ramY = 50;
int DL = 25;
int adaptiveMax = 255;
int adaptiveMask = 67;
unsigned int start_time;
Mat tresh2;
Mat input;
int thread = 0;
vector<vector<Point> > OutVecVec;
vector<Point> OutVec;


std::string toString(int val)
{
	std::ostringstream oss;
	oss << val;
	return oss.str();
}


DetectDefect::DetectDefect()
{
	
}

ReturnValue DetectDefect::GetDefect(std::string strName)
{
	printf("START");
	start_time = clock();
	std::string name = strName;

	vector<vector<Point> > squares;
	input = imread(name);

	Mat smallImage;// = input.clone();
	Mat bigImage;// = input.clone();

	//Rezkost(input, 1);

	/*resize(input, bigImage, cvSize((double)input.cols / coefBigImg, (double)input.rows / coefBigImg), 0.0, 0.0, 3);
	resize(bigImage, smallImage, cvSize(bigImage.cols / coefDel, bigImage.rows / coefDel), 0.0, 0.0, 3);*/

	//Kontrast(input, 60);

	//resize(input, bigImage, cvSize((double)input.cols / firstCoefBigImg, (double)input.rows / firstCoefBigImg), 0.0, 0.0, 3);
	printf("\n Image Size: %d x %d\n", input.cols, input.rows);
	resize(input, input, cvSize(input.cols / coefDel, input.rows / coefDel), 0.0, 0.0, 3);
	printf("\n RESIZE:");
	printf("\n Image Size after resize: %d x %d\n", input.cols, input.rows);

	unsigned int timeReadandRes1 = clock();
	unsigned int timeReadandRes2 = timeReadandRes1 - start_time;
	printf("\n TIME after resize: %d \n", timeReadandRes2);

	//findSquares(smallImage, squares);
	//drawSquaresFromSmall(smallImage, squares, input);

	unsigned int end_time = clock(); // конечное время
	unsigned int search_time = end_time - start_time;
	printf("\n TIME: %d \n", search_time);

	ReturnValue rv;
	rv.lineLength = 10;
	rv.retValue = OutVecVec;
	return rv;
}


DetectDefect::~DetectDefect()
{
}


void * Thread1(void* pParams)
{
	// 1 четверть
	for (int i = 0; i < tresh2.rows / 2; i++)
		for (int j = 0; j < tresh2.cols / 2; j++)
		{
			Rect roi(j, i, 6, 6);
			Mat sub = tresh2(roi);
			if (sub.at<uchar>(0, 0) > 10 &&
				sub.at<uchar>(0, 1) > 10 &&
				sub.at<uchar>(0, 2) > 10 &&
				sub.at<uchar>(0, 3) > 10 &&
				sub.at<uchar>(0, 4) > 10 &&
				sub.at<uchar>(0, 5) > 10 &&
				sub.at<uchar>(1, 0) > 10 &&
				sub.at<uchar>(1, 5) > 10 &&
				sub.at<uchar>(2, 0) > 10 &&
				sub.at<uchar>(2, 5) > 10 &&
				sub.at<uchar>(3, 0) > 10 &&
				sub.at<uchar>(3, 5) > 10 &&
				sub.at<uchar>(4, 0) > 10 &&
				sub.at<uchar>(4, 5) > 10 &&
				sub.at<uchar>(5, 0) > 10 &&
				sub.at<uchar>(5, 1) > 10 &&
				sub.at<uchar>(5, 2) > 10 &&
				sub.at<uchar>(5, 3) > 10 &&
				sub.at<uchar>(5, 4) > 10 &&
				sub.at<uchar>(5, 5) > 10
				)
				sub += 255;
		}
		thread++;
	printf("\nEND Thread 1");
}
void * Thread2(void* pParams)
{
	// 2 четверть
	for (int i = 0; i < tresh2.rows / 2; i++)
		for (int j = tresh2.cols / 2; j < tresh2.cols - 6; j++)
		{
			Rect roi(j, i, 6, 6);
			Mat sub = tresh2(roi);
			if (sub.at<uchar>(0, 0) > 10 &&
				sub.at<uchar>(0, 1) > 10 &&
				sub.at<uchar>(0, 2) > 10 &&
				sub.at<uchar>(0, 3) > 10 &&
				sub.at<uchar>(0, 4) > 10 &&
				sub.at<uchar>(0, 5) > 10 &&
				sub.at<uchar>(1, 0) > 10 &&
				sub.at<uchar>(1, 5) > 10 &&
				sub.at<uchar>(2, 0) > 10 &&
				sub.at<uchar>(2, 5) > 10 &&
				sub.at<uchar>(3, 0) > 10 &&
				sub.at<uchar>(3, 5) > 10 &&
				sub.at<uchar>(4, 0) > 10 &&
				sub.at<uchar>(4, 5) > 10 &&
				sub.at<uchar>(5, 0) > 10 &&
				sub.at<uchar>(5, 1) > 10 &&
				sub.at<uchar>(5, 2) > 10 &&
				sub.at<uchar>(5, 3) > 10 &&
				sub.at<uchar>(5, 4) > 10 &&
				sub.at<uchar>(5, 5) > 10
				)
				sub += 255;
		}
		thread++;
	printf("\nEND Thread 2");

}
void * Thread3(void* pParams)
{
	// 3 четверть
	for (int i = tresh2.rows / 2; i < tresh2.rows - 6; i++)
		for (int j = 0; j < tresh2.cols / 2; j++)
		{
			Rect roi(j, i, 6, 6);
			Mat sub = tresh2(roi);
			if (sub.at<uchar>(0, 0) > 10 &&
				sub.at<uchar>(0, 1) > 10 &&
				sub.at<uchar>(0, 2) > 10 &&
				sub.at<uchar>(0, 3) > 10 &&
				sub.at<uchar>(0, 4) > 10 &&
				sub.at<uchar>(0, 5) > 10 &&
				sub.at<uchar>(1, 0) > 10 &&
				sub.at<uchar>(1, 5) > 10 &&
				sub.at<uchar>(2, 0) > 10 &&
				sub.at<uchar>(2, 5) > 10 &&
				sub.at<uchar>(3, 0) > 10 &&
				sub.at<uchar>(3, 5) > 10 &&
				sub.at<uchar>(4, 0) > 10 &&
				sub.at<uchar>(4, 5) > 10 &&
				sub.at<uchar>(5, 0) > 10 &&
				sub.at<uchar>(5, 1) > 10 &&
				sub.at<uchar>(5, 2) > 10 &&
				sub.at<uchar>(5, 3) > 10 &&
				sub.at<uchar>(5, 4) > 10 &&
				sub.at<uchar>(5, 5) > 10
				)
				sub += 255;
		}
		thread++;
	printf("\nEND Thread 3");

}
void * Thread4(void* pParams)
{
	// 4 четверть
	for (int i = tresh2.rows / 2; i < tresh2.rows - 6; i++)
		for (int j = tresh2.cols / 2; j < tresh2.cols - 6; j++)
		{
			Rect roi(j, i, 6, 6);
			Mat sub = tresh2(roi);
			if (sub.at<uchar>(0, 0) > 10 &&
				sub.at<uchar>(0, 1) > 10 &&
				sub.at<uchar>(0, 2) > 10 &&
				sub.at<uchar>(0, 3) > 10 &&
				sub.at<uchar>(0, 4) > 10 &&
				sub.at<uchar>(0, 5) > 10 &&
				sub.at<uchar>(1, 0) > 10 &&
				sub.at<uchar>(1, 5) > 10 &&
				sub.at<uchar>(2, 0) > 10 &&
				sub.at<uchar>(2, 5) > 10 &&
				sub.at<uchar>(3, 0) > 10 &&
				sub.at<uchar>(3, 5) > 10 &&
				sub.at<uchar>(4, 0) > 10 &&
				sub.at<uchar>(4, 5) > 10 &&
				sub.at<uchar>(5, 0) > 10 &&
				sub.at<uchar>(5, 1) > 10 &&
				sub.at<uchar>(5, 2) > 10 &&
				sub.at<uchar>(5, 3) > 10 &&
				sub.at<uchar>(5, 4) > 10 &&
				sub.at<uchar>(5, 5) > 10
				)
				sub += 255;
		}
		thread++;
	printf("\nEND Thread 4");

}

void DetectDefect::Kontrast(Mat& src, int step = 40)
{
	printf("\nSTART Kontrast");
	//////////////////////////////Повышение контраста//////////////
	vector<Mat> rgb;
	split(src, rgb);
	Mat lut(1, 256, CV_8UC1);
	double contrastLevel = double(100 + step) / 100;
	uchar* p3 = lut.data;
	double d;
	for (int i = 0; i < 256; i++)
	{
		d = ((double(i) / 255 - 0.5)*contrastLevel + 0.5) * 255;
		if (d > 255)
			d = 255;
		if (d < 0)
			d = 0;
		p3[i] = d;
	}
	LUT(rgb[0], lut, rgb[0]);
	LUT(rgb[1], lut, rgb[1]);
	LUT(rgb[2], lut, rgb[2]);
	merge(rgb, src);
	printf("\nEND Kontrast");
	////////////////////////////////////////////////////////////////
}

void DetectDefect::Rezkost(Mat& src, float step)
{
	////////////////////////////// Резкость //////////////////////////
	//Mat rezk2(src);
	//Mat dst = src.clone();
	float a0375 = 0.0375;
	float a05 = 0.05;
	float matr[9] {
		-a0375 - a05*step, -a0375 - a05*step, -a0375 - a05*step,
			-a0375 - a05*step, 1.3 + 0.6*step, -a0375 - a05*step,
			-a0375 - a05*step, -a0375 - a05*step, -a0375 - a05*step
	};
	Mat kernel_matrix = Mat(3, 3, CV_32FC1, &matr);
	cv::filter2D(src, src, 32, kernel_matrix);
	//return src;
	////////////////////////////////////////////////////////////////////
}

void DetectDefect::DrawCntrs(Mat& src, int a = 20, int b = 255, int cl = 0)
{
	unsigned int timeReadandRes1 = clock();
	unsigned int timeReadandRes2 = timeReadandRes1 - start_time;
	printf("\n START Draw countours RED - %d", timeReadandRes2);
	Mat bin;
	Mat tresh = src.clone();
	Mat element;
	element = Mat();
	cvtColor(tresh, tresh, CV_BGR2GRAY);
	//imwrite("sources/gray.jpg", tresh);
	//Canny(src, tresh, a, b, 3);
	//threshold(tresh, bin, 200, 255, 1);
	adaptiveThreshold(tresh, tresh, adaptiveMax, 0, 0, adaptiveMask, 5);
	imwrite("tresh-dil 1.jpg", tresh);





	//imwrite("sources/tresh-dil 2.jpg", tresh);


	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	resize(tresh, tresh2, cvSize((double)tresh.cols / coefBigImg, (double)tresh.rows / coefBigImg), 0.0, 0.0, 0);
	resize(src, src, cvSize(tresh2.cols, tresh2.rows), 0.0, 0.0, 3);

	//erode(tresh2, tresh2, Mat(), Point(-1, -1), 1, 1, 1);
	//dilate(tresh2, tresh2, Mat(), Point(-1, -1), 1, 1, 1);

	//imwrite("sources/tresh-dil 3.jpg", tresh2);


	//imwrite("sources/tresh-dil 4.jpg", tresh2);



	/*for (int i = 0; i < tresh2.rows - 4; i++)
	for (int j = 0; j < tresh2.cols - 4; j++)
	{
	Rect roi(j, i, 4, 4);
	Mat sub = tresh2(roi);
	if (sub.at<uchar>(0, 0) > 10 &&
	sub.at<uchar>(0, 1) > 10 &&
	sub.at<uchar>(0, 2) > 10 &&
	sub.at<uchar>(0, 3) > 10 &&
	sub.at<uchar>(1, 0) > 10 &&
	sub.at<uchar>(1, 3) > 10 &&
	sub.at<uchar>(2, 0) > 10 &&
	sub.at<uchar>(2, 3) > 10 &&
	sub.at<uchar>(3, 0) > 10 &&
	sub.at<uchar>(3, 1) > 10 &&
	sub.at<uchar>(3, 2) > 10 &&
	sub.at<uchar>(3, 3) > 10
	)
	sub += 255;
	}*/
	/*pthread_t thread;
	pthread_t thread;
	pthread_t thread;
	pthread_t thread;
	printf("START Threads");*/

	pthread_t thread1;
	pthread_t thread2;
	pthread_t thread3;
	pthread_t thread4;
	printf("\nSTART Threads");
	pthread_create(&thread1, NULL, &Thread1, NULL);
	pthread_create(&thread2, NULL, &Thread2, NULL);
	pthread_create(&thread3, NULL, &Thread3, NULL);
	pthread_create(&thread4, NULL, &Thread4, NULL);

	// while (thread != 4)
	// {
	// }
	pthread_join(thread1, NULL);
	pthread_join(thread2, NULL);
	pthread_join(thread3, NULL);
	pthread_join(thread4, NULL);
	printf("END Threads");

	//erode(tresh2, tresh2, Mat(), Point(-1, -1), 1, 1, 1);
	//imwrite("sources/tresh-dil 2.jpg", tresh2);

	/*for (int i = 0; i < tresh2.rows - 10; i++)
	for (int j = 0; j < tresh2.cols - 10; j++)
	{
	Rect roi(j, i, 10, 10);
	Mat sub = tresh2(roi);
	if (sub.at<uchar>(0, 0) > 10 &&
	sub.at<uchar>(0, 1) > 10 &&
	sub.at<uchar>(0, 2) > 10 &&
	sub.at<uchar>(0, 3) > 10 &&
	sub.at<uchar>(0, 4) > 10 &&
	sub.at<uchar>(0, 5) > 10 &&
	sub.at<uchar>(0, 6) > 10 &&
	sub.at<uchar>(0, 7) > 10 &&
	sub.at<uchar>(0, 8) > 10 &&
	sub.at<uchar>(0, 9) > 10 &&
	sub.at<uchar>(1, 0) > 10 &&
	sub.at<uchar>(1, 9) > 10 &&
	sub.at<uchar>(2, 0) > 10 &&
	sub.at<uchar>(2, 9) > 10 &&
	sub.at<uchar>(3, 0) > 10 &&
	sub.at<uchar>(3, 9) > 10 &&
	sub.at<uchar>(4, 0) > 10 &&
	sub.at<uchar>(4, 9) > 10 &&
	sub.at<uchar>(5, 0) > 10 &&
	sub.at<uchar>(5, 9) > 10 &&
	sub.at<uchar>(6, 0) > 10 &&
	sub.at<uchar>(6, 9) > 10 &&
	sub.at<uchar>(7, 0) > 10 &&
	sub.at<uchar>(7, 9) > 10 &&
	sub.at<uchar>(8, 0) > 10 &&
	sub.at<uchar>(8, 9) > 10 &&
	sub.at<uchar>(9, 0) > 10 &&
	sub.at<uchar>(9, 1) > 10 &&
	sub.at<uchar>(9, 2) > 10 &&
	sub.at<uchar>(9, 3) > 10 &&
	sub.at<uchar>(9, 4) > 10 &&
	sub.at<uchar>(9, 5) > 10 &&
	sub.at<uchar>(9, 6) > 10 &&
	sub.at<uchar>(9, 7) > 10 &&
	sub.at<uchar>(9, 8) > 10 &&
	sub.at<uchar>(9, 9) > 10
	)
	sub += 255;
	}*/

	//imwrite("sources/beforeDraw.jpg", tresh2);
	
	

	findContours(tresh2, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	printf("\ncontours are found - %d", contours.size());

	Mat src2 = src.clone();
	src2 *= 255;
	//cvtColor(src2, src2, CV_BGR2GRAY);
	//imwrite("WaR1.jpg", src2);
	//Рисуем найденные картинки с Кенни контуры на цветной картинке 
	for (int i = 0; i< contours.size(); i++)
	{
		if (contours[i].size() < cl)
			continue;
		//printf("iter %d", i);
		Scalar color = Scalar(0, 0, 255);
		drawContours(src2, contours, i, color, 2, 8, hierarchy, 0, Point());
	}
	
	blur(src2, src2, Size(5, 5));

	resize(src2, src2, cvSize(src2.cols / coefAfterTresh, src2.rows / coefAfterTresh), 0.0, 0.0, 3);
	imwrite("WaR2.jpg", src2);
	src2 = DrawLargeCntrs(src2, 0, 180, 25);
	
	unsigned int timeReadandRes3 = clock();
	unsigned int timeReadandRes4 = timeReadandRes3 - start_time;
	printf("\n END Draw countours RED - %d", timeReadandRes4);
}

Mat DetectDefect::DrawLargeCntrs(Mat& src, int a = 0, int b = 255, int cl = 100)
{
	printf("\n START Draw LARGE countours GREEN");
	Mat bin;
	//Mat out(src);
	Canny(src, bin, a, b, 3);
	long contourLength = 0;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	
	Mat out;
	Rect r(ramX, ramY, bin.cols - 55, bin.rows - 55);
	bin(r).copyTo(out);
	//imwrite("sources/before.jpg", out);

	findContours(out, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	printf("contours are found - %d", contours.size());

	cvtColor(out, out, CV_GRAY2BGR);
	vector<vector<Point> > contours2;
	//Рисуем найденные картинки с Кенни контуры на цветной картинке 
	for (int i = 0; i< contours.size(); i++)
	{
		if (contours[i].size() < cl)
			continue;
		//printf("iter %d", i);
		Scalar color = Scalar(rand() % 255, rand() % 255, rand() % 255);
		drawContours(out, contours, i, color, 1, 8, hierarchy, 0, Point());
		contourLength += contours[i].size();
		contours2.push_back(contours[i]);
	}

	for (int i = 0; i< contours2.size(); i++)
	{
		//printf("iter %d", i);
		Scalar color = Scalar(0, 250, 0);
		drawContours(out, contours2, i, color, 1, 8, NULL, 0, Point());
	}

	for (size_t i = 0; i < contours2.size(); i++)
	{
		for (size_t j = 0; j < contours2[i].size(); j++)
		{
			Point pp = contours2[i][j];
			pp.x = ((pp.x + ramX) * coefAfterTresh * coefBigImg + wallXmin) * firstCoefBigImg;
			pp.y = ((pp.y + ramY) * coefAfterTresh * coefBigImg + wallYmin) * firstCoefBigImg;
			OutVec.push_back(pp);
		}
		OutVecVec.push_back(OutVec);
	}
	


	for (int i = 0; i< OutVecVec.size(); i++)
	{
		//printf("iter %d", i);
		Scalar color = Scalar(0, 250, 0);
		drawContours(input, OutVecVec, i, color, 4, 8, NULL, 0, Point());
	}


	//////////////////////////////////////// Формирование текста длины линии дефекта//////////////
	/*Point p4 = Point(src.cols / 2 - 500, 100);
	Rect r4(src.cols / 2 - 500, 0, 1500, 150);
	src(r4) *= 0.3;
	std::string str = "Length of Defect -  " + toString(contourLength / 2) + "px";
	putText(src, str, p4, 2, 3.0, cv::Scalar(230, 230, 230), 4);*/
	//////////////////////////////////////////////////////////////////////////////////////////////////

	//namedWindow("lc", 1);
	//imshow("lc", outPut);
	imwrite("input.jpg", input);
	//printf("Defect length - %d", contourLength/2);

	printf("\n END Draw LARGE countours GREEN");

	return out;
}

Mat DetectDefect::DrawDefects(Mat& canny2)
{
	int KON = 60;
	float REZ = 0.5;
	int CA = 0;
	int CB = 255;
	Mat out2;
	printf("\n START draw Defect");

	//Rezkost(canny2, REZ);
	Rezkost(canny2, REZ);
	//Kontrast(canny2, KON);
	//Kontrast(canny2, KON);
	DrawCntrs(canny2, CA, CB, DL);

	printf("\n END draw Defect");
	return canny2;
}

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
double DetectDefect::angle(Point pt1, Point pt2, Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
void DetectDefect::findSquares(Mat& image, vector<vector<Point> >& squares)
{
	printf("\n START find Squares");
	squares.clear();
	Mat pyr, timg, gray0(image.size(), CV_8U), gray;
	// down-scale and upscale the image to filter out the noise
	pyrDown(image, pyr, Size(image.cols / 2, image.rows / 2));
	pyrUp(pyr, timg, image.size());
	vector<vector<Point> > contours;

	// find squares in every color plane of the image
	for (int c = 0; c < 3; c++)
	{
		int ch[] = { c, 0 };
		mixChannels(&timg, 1, &gray0, 1, ch, 1);

		// try several threshold levels
		for (int l = 0; l < N; l++)
		{
			// hack: use Canny instead of zero threshold level.
			// Canny helps to catch squares with gradient shading
			if (l == 0)
			{
				// apply Canny. Take the upper threshold from slider
				// and set the lower to 0 (which forces edges merging)
				Canny(gray0, gray, 0, thresh, 5);
				// dilate canny output to remove potential
				// holes between edge segments
				dilate(gray, gray, Mat(), Point(-1, -1));
			}
			else
			{
				// apply threshold if l!=0:
				//     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
				gray = gray0 >= (l + 1) * 255 / N;
			}

			// find contours and store them all as a list
			findContours(gray, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

			vector<Point> approx;

			// test each contour
			for (size_t i = 0; i < contours.size(); i++)
			{
				// approximate contour with accuracy proportional
				// to the contour perimeter
				approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

				// square contours should have 4 vertices after approximation
				// relatively large area (to filter out noisy contours)
				// and be convex.
				// Note: absolute value of an area is used because
				// area may be positive or negative - in accordance with the
				// contour orientation
				if (approx.size() == 4 &&
					fabs(contourArea(Mat(approx))) > 1000 &&
					isContourConvex(Mat(approx)))
				{
					double maxCosine = 0;

					for (int j = 2; j < 5; j++)
					{
						// find the maximum cosine of the angle between joint edges
						double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
						maxCosine = MAX(maxCosine, cosine);
					}

					// if cosines of all angles are small
					// (all angles are ~90 degree) then write quandrange
					// vertices to resultant sequence
					//if (maxCosine < coefCosine)
					if (maxCosine < coefCosine)
						squares.push_back(approx);
				}
			}
		}
	}
	unsigned int timeReadandRes1 = clock();
	unsigned int timeReadandRes2 = timeReadandRes1 - start_time;
	printf("\n END find Squares - %d", timeReadandRes2);
}

void DetectDefect::drawSquaresFromSmall(Mat& image, vector<vector<Point> >& squares, Mat& bigImage)
{
	size_t index = 0;
	int max = 0;

	////Находим квадрат у которого левый верхний угол самый крайний (ближе к правой стороне)////////
	for (size_t i = 0; i < squares.size(); i++)
	{
		const Point* p = &squares[i][0];
		if (p->x > max)
		{
			max = p->x;
			index = i;
		}
	}

	vector<vector<Point> > vecVec;
	vector<Point> vec;
	Point pp;

	for (size_t i = 0; i < 4; i++)
	{
		Point pp = squares[index][i];
		pp.x = pp.x * coefDel;
		pp.y = pp.y * coefDel;
		vec.push_back(pp);
	}
	vecVec.push_back(vec);

	int nn = (int)vecVec[0].size();
	const Point* pp1 = &vecVec[0][0];
	polylines(bigImage, &pp1, &nn, 1, true, Scalar(255, 0, 0), 10, CV_AA);

	//imwrite("sources/outputBig.jpg", bigImage);


	///////////Строим квадрат, чуть изменяя координаты найденного квадрата///////////////////////
	const Point* p = &vecVec[0][0];
	int xmin = 10000, ymin = 10000, xmax = 0, ymax = 0;
	for (int i = 0; i < 4; i++)
	{
		if (xmax < p[i].x)
			xmax = p[i].x;

		if (ymax < p[i].y)
			ymax = p[i].y;

		if (xmin > p[i].x)
			xmin = p[i].x;

		if (ymin > p[i].y)
			ymin = p[i].y;
	}
	//Rect r(xmin + 100, ymin + 200, xmax - xmin - 300, ymax - ymin - 400);
	//Rect r(xmin + 300, ymin + 500, xmax - xmin - 600, ymax - ymin - 1000);
	//Rect r(xmin + 200, ymin + 200, xmax - xmin - 400, ymax - ymin - 500);
	Rect r(xmin - wallXmin, ymin + wallYmin, xmax - xmin - wallXmax, ymax - ymin - wallYmax);
	wallXmin = xmin - wallXmin;
	wallYmin = ymin + wallYmin;
	Mat canny2;
	bigImage(r).copyTo(canny2);
	imwrite("outputBigWall.jpg", canny2);

	Mat predOut = DrawDefects(canny2);
	//Mat out = DrawLargeCntrs(predOut, 0, 255, 100);

	printf("\nSave step");
	//imwrite("OutLastR5.jpg", predOut);
	printf("\nthats all");
}

