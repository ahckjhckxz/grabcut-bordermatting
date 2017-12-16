#pragma once
#include "stdafx.h"
#include "GMM.h"
using namespace cv;
enum
{
	GC_WITH_RECT  = 0, 
	GC_WITH_MASK  = 1, 
	GC_CUT        = 2  
};
class GrabCut2D
{
public:
	void GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect,
		cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel,
		int iterCount, int mode );  
	double CalBeta(const Mat img);
	void BoardMatting();
	Mat src;
	Mat mask;
	void initTrimap(Mat mask, Rect rect);
	void CalV(const Mat img, double beta, Mat left, Mat upleft, Mat up, Mat upright);
	void ConstructGraph(const Mat img, Mat mask, Mat left, Mat upleft, Mat up, Mat upright, GMM& bgd, GMM& fgd, GCGraph<double> &graph);
	~GrabCut2D(void);
};

