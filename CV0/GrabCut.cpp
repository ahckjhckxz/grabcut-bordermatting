//һ.�������ͣ�
//���룺
//cv::InputArray _img,     :�����colorͼ��(����-cv:Mat)
//cv::Rect rect            :��ͼ���ϻ��ľ��ο�����-cv:Rect) 
//int iterCount :           :ÿ�ηָ�ĵ�������������-int)


//�м����
//cv::InputOutputArray _bgdModel ��   ����ģ�ͣ��Ƽ�GMM)������-13*n�������������double���͵��Զ������ݽṹ������Ϊcv:Mat������Vector/List/����ȣ�
//cv::InputOutputArray _fgdModel :    ǰ��ģ�ͣ��Ƽ�GMM) ������-13*n�������������double���͵��Զ������ݽṹ������Ϊcv:Mat������Vector/List/����ȣ�


//���:
//cv::InputOutputArray _mask  : ����ķָ��� (���ͣ� cv::Mat)

//��. α�������̣�
//1.Load Input Image: ����������ɫͼ��;
//2.Init Mask: �þ��ο��ʼ��Mask��Labelֵ��ȷ��������0�� ȷ��ǰ����1�����ܱ�����2������ǰ����3��,���ο���������Ϊȷ�����������ο���������Ϊ����ǰ��;
//3.Init GMM: ���岢��ʼ��GMM(����ģ����ɷָ�Ҳ�ɵõ�����������GMM��ɻ�ӷ֣�
//4.Sample Points:ǰ������ɫ���������о��ࣨ������kmeans���������෽��Ҳ��)
//5.Learn GMM(���ݾ������������ÿ��GMM����еľ�ֵ��Э����Ȳ�����
//4.Construct Graph������t-weight(�������n-weight��ƽ�����
//7.Estimate Segmentation(����maxFlow����зָ�)
//8.Save Result�������������mask�������mask��ǰ�������Ӧ�Ĳ�ɫͼ�񱣴����ʾ�ڽ��������У�

#include "GrabCut.h"
#include "GMM.h"
#include "mybm.h"
using namespace cv;
const double gamma = 50;
const double lamda = gamma * 9;
mybm inst;
GrabCut2D::~GrabCut2D(void)
{
}
void GrabCut2D::initTrimap(Mat mask, Rect rect)
{
	for(int y=0; y<mask.rows; y++)
		for (int x = 0; x < mask.cols; x++)
		{
			if (x < rect.tl().x || x > rect.tl().x + rect.width || y < rect.tl().y || y > rect.tl().y + rect.height)
				mask.at<uchar>(y, x) = GC_BGD;
			else
				mask.at<uchar>(y, x) = GC_PR_FGD;
		}
}
double GrabCut2D::CalBeta(const Mat img)
{
	double beta = 0;
	for(int y=0; y<img.rows; y++)
		for (int x = 0; x < img.cols; x++)
		{
			Vec3b Point = img.at<Vec3b>(y, x);
			if (x - 1 >= 0)
			{
				Vec3b Left = img.at<Vec3b>(y, x - 1);
				beta += (Point - Left).dot(Point - Left);
			}
			if (x - 1 >= 0 && y - 1 >= 0)
			{
				Vec3b UpLeft = img.at<Vec3b>(y - 1, x - 1);
				beta += (Point - UpLeft).dot(Point - UpLeft);
			}
			if (y - 1 >= 0)
			{
				Vec3b Up = img.at<Vec3b>(y - 1, x);
				beta += (Point - Up).dot(Point - Up);
			}
			if (y - 1 >= 0 && x + 1 < img.cols)
			{
				Vec3b UpRight = img.at<Vec3b>(y - 1, x + 1);
				beta += (Point - UpRight).dot(Point - UpRight);
			}
		}
	if(beta > 0)
		beta = 1.0 / (2 * beta / (4 * img.cols*img.rows - 3 * img.cols - 3 * img.rows + 2));
	return beta;
}
void GrabCut2D::CalV(Mat img, double beta, Mat left, Mat upleft, Mat up, Mat upright)
{
	double ddga = sqrt(2)*gamma;
	for (int y = 0; y<img.rows; y++)
		for (int x = 0; x < img.cols; x++)
		{
			Vec3b Point = img.at<Vec3b>(y, x);
			if (x - 1 >= 0)
			{
				Vec3b Left = img.at<Vec3b>(y, x - 1);
				left.at <double>(y, x) = gamma * exp(-beta*(Point-Left).dot(Point - Left));
			}
			if (x - 1 >= 0 && y - 1 >= 0)
			{
				Vec3b UpLeft = img.at<Vec3b>(y - 1, x - 1);
				upleft.at <double>(y, x) = ddga * exp(-beta*(Point - UpLeft).dot(Point - UpLeft));
			}
			if (y - 1 >= 0)
			{
				Vec3b Up = img.at<Vec3b>(y - 1, x);
				up.at <double>(y, x) = gamma * exp(-beta*(Point - Up).dot(Point - Up));
			}
			if (y - 1 >= 0 && x + 1 < img.cols)
			{
				Vec3b UpRight = img.at<Vec3b>(y - 1, x + 1);
				upright.at <double>(y, x) = ddga * exp(-beta*(Point - UpRight).dot(Point - UpRight));
			}
		}
}
void GrabCut2D::ConstructGraph(const Mat img, Mat mask, Mat left, Mat upleft, Mat up, Mat upright, GMM& bgd, GMM& fgd, GCGraph<double> &graph)
{
	for(int y=0; y<img.rows; y++)
		for (int x = 0; x < img.cols; x++)
		{
			int NodeID = graph.addVtx();
			Vec3b Point = img.at<Vec3b>(y, x);
			double WeigtL = left.at<double>(y, x);
			double WeigtUL = upleft.at<double>(y, x);
			double WeigtU = up.at<double>(y, x);
			double WeigtUR = upright.at<double>(y, x);
			double S, T;
			if (mask.at<uchar>(y, x) == GC_PR_BGD || mask.at<uchar>(y, x) == GC_PR_FGD)
			{
				double tmp[3] = { Point[0], Point[1], Point[2] };
				S = -log(bgd.GetProbability(tmp));
				T = -log(fgd.GetProbability(tmp));
			}
			else if(mask.at<uchar>(y, x) == GC_BGD)
			{
				S = 0;
				T = lamda;
			}
			else
			{
				S = lamda;
				T = 0;
			}
			graph.addTermWeights(NodeID, S, T);
			if (x-1>=0)
				graph.addEdges(NodeID, NodeID - 1, WeigtL, WeigtL);
			if (x-1>=0 && y -1 >=0)
				graph.addEdges(NodeID, NodeID - img.cols - 1, WeigtUL, WeigtUL);
			if (y-1>=0)
				graph.addEdges(NodeID, NodeID - img.cols, WeigtU, WeigtU);
			if (x + 1 < img.cols && y-1>=0)
				graph.addEdges(NodeID, NodeID - img.cols + 1, WeigtUR, WeigtUR);
		}
}
void GrabCut2D::GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect, cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel, int iterCount, int mode )
{
	src = _img.getMat();
	Mat left = Mat::zeros(src.size(), CV_64F);
	Mat upleft = Mat::zeros(src.size(), CV_64F);
	Mat up = Mat::zeros(src.size(), CV_64F);
	Mat upright = Mat::zeros(src.size(), CV_64F);
	double beta = CalBeta(src);
	switch (mode)
	{
		case GC_WITH_RECT:
		{
			mask = Mat::zeros(src.size(), CV_8U);
			initTrimap(mask, rect);
			mask = _mask.getMat();
			break;
		}
		case GC_WITH_MASK:
		{
			mask = _mask.getMat();
			break;
		}
		case GC_CUT:
		{
			mask = _mask.getMat();
			break;
		}
	}
	int VertexCount = src.cols*src.rows;
	int EdgeCount = 2 * (4 * VertexCount - 3 * (src.cols + src.rows) + 2);
	for (int itrcnt = 0; itrcnt < iterCount; itrcnt++)
	{
	
		GMM bgdGMM, fgdGMM;
		vector<double> bgd, fgd;
		GCGraph<double> graph;
		graph.create(VertexCount, EdgeCount);
		for (int y = 0; y<mask.rows; y++)
			for (int x = 0; x < mask.cols; x++)
			{
				Vec3b tmp = src.at<Vec3b>(y, x);
				if (mask.at<uchar>(y, x))
				{
					fgd.push_back(tmp[0]);
					fgd.push_back(tmp[1]);
					fgd.push_back(tmp[2]);
				}
				else
				{
					bgd.push_back(tmp[0]);
					bgd.push_back(tmp[1]);
					bgd.push_back(tmp[2]);
				}
			}
		double *data = new double[fgd.size()];
		for (int i = 0; i < fgd.size(); i++)
			data[i] = fgd[i];
		fgdGMM.Train(data, fgd.size() / 3);
		delete[]data;
		data = new double[bgd.size()];
		for (int i = 0; i < bgd.size(); i++)
			data[i] = bgd[i];
		bgdGMM.Train(data, bgd.size() / 3);
		CalV(src, beta, left, upleft, up, upright);
		ConstructGraph(src, mask, left, upleft, up, upright, bgdGMM, fgdGMM, graph);
		graph.maxFlow();
		for (int y = 0; y<mask.rows; y++)
			for (int x = 0; x < mask.cols; x++)
			{
				if (mask.at<uchar>(y, x) != GC_FGD)
				{
					if (graph.inSourceSegment(y*mask.cols + x))
						mask.at<uchar>(y, x) = GC_PR_FGD;
					else
						mask.at<uchar>(y, x) = GC_BGD;
				}
			}
	}
}
void GrabCut2D::BoardMatting()
{
	Mat rst = Mat(src.size(), src.type());
	src.copyTo(rst);
	for (int i = 0; i<rst.rows; i++)
		for (int j = 0; j < rst.cols; j++)
		{
			if (mask.at<uchar>(i, j) == 0)
				rst.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
		}
	imwrite("����G.jpg", rst);
	inst.Init(src, mask);
	inst.Run();
}