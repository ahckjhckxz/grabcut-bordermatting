#include "mybm.h"
using namespace cv;


mybm::mybm(const Mat& _originImage, const Mat& _mask)
{
	Init(_originImage, _mask);
}
mybm::mybm()
{
}
mybm::~mybm()
{
}
double mybm::linear(double x, double _delta, double _sigma) {
	double k = 0;
	if (_sigma >= 0.1)
		k = 1 / _sigma;
	if (x < _delta - _sigma / 2)
		return 0;
	if (x >= _delta + _sigma / 2)
		return 1;
	return 0.5 + k*(x - _delta);
}
void mybm::Init(const Mat& _originImage, const Mat& _mask)
{
	for (int i = 0; i<6; i++)
		for (int j = 0; j<30; j++)
			for (int k = 0; k < 10; k++)
			{
				g[i][j][k] = linear(i, j*0.2, k*0.6);
			}
	_mask.copyTo(Mask);
	Mask = Mask & 1;
	_originImage.copyTo(Src);
	Canny(Mask, Edge, 1, 5, 3);   //边缘提取
	ConstructContour();
	ConstructStrip();
}
void mybm::ConstructStrip()
{
	for(int i=0; i<Src.rows; i++)
		for (int j = 0; j < Src.cols; j++)
		{
			if (Edge.at<uchar>(i, j) == 0 && Mask.at<uchar>(i, j) == 1)
			{
				point p(j, i);
				int dis = CheckDis(p);
				if (dis > 0)
				{
					Push(p, 3);
				}
			}
		}
}
void mybm::ConstructContour()
{
	for(int i=0; i<Src.rows; i++)
		for (int j = 0; j < Src.cols; j++)
		{
			if (Edge.at<uchar>(i, j))
			{
				Push(point(j, i), contour, 2);
			}
		}
}
int mybm::CheckDis(point p)
{
	for (int i = 0; i < contour.size(); i++)
	{
		if (p.distance(contour[i].p) < 3)
			return p.distance(contour[i].p);
	}
	return -1;
}
void mybm::Push(point p, vector<Contour> &list, int threshold)
{
	int min = -1;
	for (int i = 0; i < list.size(); i++)
	{
		if (p.distance(list[i].p) <= threshold)
		{
			list.insert(list.begin() + i, Contour(p));
			return;
		}
	}
	list.push_back(Contour(p));
}
void mybm::Push(point p, int threshold)
{
	int min = -1;
	for (int i = 0; i < contour.size(); i++)
	{
		if (p.distance(contour[i].p) <= threshold)
		{
			if (p.distance(contour[i].p) < threshold)
			{
				min = i;
				threshold = p.distance(contour[i].p);
			}
		}
	}
	
	if (min == -1)
		return;
	else
	{
		p.dis = threshold;
		contour[min].neighbor.push_back(p);
	}
}
void mybm::getLocalMandV(point p, LocalPara &rst)
{
	int x = (p.p.x - 20 < 0) ? 0 : p.p.x - 20;
	int xlen = (x + 41 < Src.cols) ? 41 : Src.cols - x;
	int y = (p.p.y - 20 < 0) ? 0 : p.p.y - 20;
	int ylen = (y + 41 < Src.rows) ? 41 : Src.rows - y;
	Mat neibor = Src(Rect(x,y, xlen, ylen));
	Vec3i Bmean, Fmean;
	double Bvar = 0, Fvar = 0;
	int Fn=0, Bn=0;
	for(int i=0; i<neibor.rows; i++)
		for (int j = 0; j < neibor.cols; j++)
		{
			Vec3i tmp(neibor.at<Vec3b>(i, j)[0], neibor.at<Vec3b>(i, j)[1], neibor.at<Vec3b>(i, j)[2]);
			if (Edge.at<uchar>(y + i, x + j) == 1)
			{
				Fmean += tmp;
				Fn++;
			}
			else
			{
				Bmean += tmp;;
				Bn++;
			}			
		}
	if (Fn)
		Fmean = Fmean / Fn;
	else
		Fmean = 0;
	if (Bn)
		Bmean = Bmean / Bn;
	else
		Bmean = 0;
	for (int i = 0; i<neibor.rows; i++)
		for (int j = 0; j < neibor.cols; j++)
		{
			Vec3i tmp(neibor.at<Vec3b>(i, j)[0], neibor.at<Vec3b>(i, j)[1], neibor.at<Vec3b>(i, j)[2]);
			if (Edge.at<uchar>(y + i, x + j) == 1)
				Fvar += (Fmean - tmp).dot(Fmean - tmp);
			else
				Bvar += (tmp - Bmean).dot(tmp - Bmean);
		}
	if (Fn)
		Fvar = Fvar / Fn;
	else
		Fvar = 0;
	if (Bn)
		Bvar = Bvar / Bn;
	else
		Bvar = 0;
	rst.Bmean = Bmean;
	rst.Bvar = Bvar;
	rst.Fmean = Fmean;
	rst.Fvar = Fvar;
}
double mybm::Gaussian(double _x, double _delta, double _sigma) {
	const double PI = 3.14159;
	double e = exp(-(pow(_x - _delta, 2.0) / (2.0*_sigma)));
	double rs = 1.0 / (pow(_sigma, 0.5)*pow(2.0*PI, 0.5))*e;
	return rs;
}
//论文中公式15（1）
double mybm::Mmean(double x, double Fmean, double Bmean) {
	return (1.0 - x)*Bmean + x*Fmean;
}
//论文中公式15（2）
double mybm::Mvar(double x, double Fvar, double Bvar) {
	return (1.0 - x)*(1.0 - x)*Bvar + x*x*Fvar;
}
double mybm::dataTermPoint(point _ip, uchar _I, int _delta, int _sigma, LocalPara &para) {
	double alpha = g[_ip.dis][_delta][_sigma];
	double D = Gaussian(_I, Mmean(alpha, toGray(para.Fmean), toGray(para.Bmean)), Mvar(alpha, para.Fvar, para.Bvar));
	D = -log(D) / log(2.0);
	return D;
}
const __int64 __NaN = 0xFFF8000000000000;
void mybm::Run()
{
	double Emin = *((double *)&__NaN);
	int delta = 15, sigma = 5;
	for(int di = 0; di < 30; di++)
		for (int si = 0; si < 10; si++)
		{
			double D = dataTermPoint(contour[0].p, toGray(Src.at<Vec3b>(contour[0].p.p.y, contour[0].p.p.x)), di, si, contour[0].p.para);
			for (int j = 0; j < contour[0].neighbor.size(); j++)
			{
				point &p = contour[0].neighbor[j];
				getLocalMandV(p, p.para);
				D += dataTermPoint(p, toGray(Src.at<Vec3b>(p.p.y, p.p.x)), di, si, p.para);
			}
			if (D < Emin)
			{
				Emin = D;
				delta = di;
				sigma = si;
			}
		}

			for (int i = 1; i < contour.size(); i++)
			{
				LocalPara para;
				getLocalMandV(contour[i].p, para);
				contour[i].p.para = para;
				for (int j = 0; j < contour[i].neighbor.size(); j++)
				{
					point &p = contour[i].neighbor[j];
					getLocalMandV(p, para);
					p.para = para;
				}
				double min = 999999999;
				int ns, nd;
				for (int si = 0; si < 30; si++)
					for (int di = 0; di < 10; di++)
					{
						double D = dataTermPoint(contour[i].p, toGray(Src.at<Vec3b>(contour[i].p.p.y, contour[i].p.p.x)), si, di, contour[i].p.para);
						for (int j = 0; j < contour[i].neighbor.size(); j++)
						{
							point &p = contour[i].neighbor[j];
							D += dataTermPoint(p, toGray(Src.at<Vec3b>(p.p.y, p.p.x)), si, di, p.para);
						}
						double V = 2 * (si - delta)*(si - delta) + 360 * (sigma - di)*(sigma - di);
						if (D + V < min)
						{
							min = D + V;
							contour[i].p.delta = si;
							contour[i].p.sigma = di;
						}
					}
				sigma = contour[i].p.sigma;
				delta = contour[i].p.delta;
				contour[i].p.alpha = g[0][delta][sigma];
				for (int j = 0; j < contour[i].neighbor.size(); j++)
				{
					point &p = contour[i].neighbor[j];
					p.alpha = g[p.dis][delta][sigma];
				}
			}
	Mat _alphaMask = Mat(Mask.size(), CV_32FC1, Scalar(0));
	for (int i = 0; i < Mask.rows; i++)
		for (int j = 0; j < Mask.cols; j++)
			_alphaMask.at<float>(i, j) = Mask.at<uchar>(i, j);
	for (int i = 0; i < contour.size(); i++)
	{
		_alphaMask.at<float>(contour[i].p.p.y, contour[i].p.p.x) = contour[i].p.alpha;
		for (int j = 0; j < contour[i].neighbor.size(); j++)
		{
			point &p = contour[i].neighbor[j];
			_alphaMask.at<float>(p.p.y, p.p.x) = p.alpha;
		}
	}

	Mat rst = Mat(Src.size(), CV_8UC4);
	for (int i = 0; i<rst.rows; i++)
		for (int j = 0; j < rst.cols; j++)
		{
			rst.at<Vec4b>(i, j) = Vec4b(Src.at<Vec3b>(i, j)[0], Src.at<Vec3b>(i, j)[1], Src.at<Vec3b>(i, j)[2], _alphaMask.at<float>(i, j)*255);
		}
	vector<int> compression_params;   compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	imwrite("骆驼B.png", rst, compression_params);
	std::cout << "done!" << std::endl;
}