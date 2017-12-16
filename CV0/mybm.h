#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
using namespace cv;
class mybm
{
private:
	Mat Mask;
	Mat Src;
	Mat Edge;
	double g[6][30][10];
public:
	mybm();
	double linear(double _r, double _delta, double _sigma);
	double Gaussian(double _x, double _delta, double _sigma);
	double Mmean(double x, double Fmean, double Bmean);
	double Mvar(double x, double Fvar, double Bvar);
	mybm(const Mat& _originImage, const Mat& _mask);
	struct LocalPara {
		Vec3b Bmean;
		Vec3b Fmean;
		double Bvar;
		double Fvar;
	};
	class  point{
	public:
		Point2d p;
		int delta, sigma;
		int dis;
		double alpha;
		LocalPara para;
		point(int x, int y, int dis=0)
		{
			this->p = Point2d(x, y);
			this->dis = dis;
		}
		point() {};
		int distance(point t)
		{
			return sqrt((this->p.x - t.p.x)*(this->p.x - t.p.x) + (this->p.y - t.p.y)*(this->p.y - t.p.y));
		}
	};
	class Contour {
	public:
		point p;
		vector<point> neighbor;
		Contour(point p) { this->p = p; };
	};
	uchar toGray(Vec3i tmp)
	{
		return (tmp[2] * 299 + tmp[1] * 587 + tmp[0] * 114 + 500) / 1000;
	}
	uchar toGray(Vec3b tmp)
	{
		return (tmp[2] * 299 + tmp[1] * 587 + tmp[0] * 114 + 500) / 1000;
	}
	void mybm::Push(point p, vector<Contour> &list, int threshold);
	void getLocalMandV(point p, LocalPara &rst);
	vector<Contour> contour;
	void Push(point p, int threshold);
	vector<Contour> Oldcontour;
	void ConstructStrip();
	double mybm::dataTermPoint(point _ip, uchar _I, int _delta, int _sigma, LocalPara &para);
	void Init(const Mat& _originImage, const Mat& _mask);
	void Run();
	void Push(point p, vector<point> &list, int threshold);
	int CheckDis(point p);
	void ConstructContour();
	~mybm();
};

