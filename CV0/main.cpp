#include "GCApplication.h"
#include <windows.h>
static void help()
{
	cout << "\nThis program demonstrates GrabCut segmentation -- select an object in a region\n"
		"and then grabcut will attempt to segment it out.\n"
		"Call:\n"
		"./grabcut <image_name>\n"
		"\nSelect a rectangular area around the object you want to segment\n" <<
		"\nHot keys: \n"
		"\tESC - quit the program\n"
		"\tr - restore the original image\n"
		"\tn - next iteration\n"
		"\n"
		"\tleft mouse button - set rectangle\n"
		"\n"
		"\tCTRL+left mouse button - set GC_BGD pixels\n"
		"\tSHIFT+left mouse button - set CG_FGD pixels\n"
		"\n"
		"\tCTRL+right mouse button - set GC_PR_BGD pixels\n"
		"\tSHIFT+right mouse button - set CG_PR_FGD pixels\n" << endl;
}


GCApplication gcapp;

static void on_mouse(int event, int x, int y, int flags, void* param)
{
	gcapp.mouseClick(event, x, y, flags, param);
}

int main(int argc, char** argv)
{
	//��ȡͼƬ�ļ�
	string filename = "����.jpg";
	if (filename.empty())
	{
		cout << "\nDurn, couldn't read any file." << endl;
		return 1;
	}
	Mat image = imread(filename, 1);
	if (image.empty())
	{
		cout << "\n Durn, couldn't read image filename " << filename << endl;
		return 1;
	}
	//����˵��
	help();

	const string winName = "image";
	namedWindow(winName, WINDOW_AUTOSIZE);
	//���������Ӧ����
	setMouseCallback(winName, on_mouse, 0);
	//��ʼ�����ں�ͼƬ
	gcapp.setImageAndWinName(image, winName);
	gcapp.showImage();
	int iterCount = 0;
	int newIterCount = 0;
	for (;;)
	{
		int c = waitKey(0);
		switch ((char)c)
		{
			//ESC�����˳�
		case '\x1b':
			cout << "Exiting ..." << endl;
			goto exit_main;
			//r��������ͼ��
		case 'r':
			cout << endl;
			gcapp.reset();
			gcapp.showImage();
			break;
			//n��������һ�δ���
		case 'n':
		{
			iterCount = gcapp.getIterCount();
			cout << "<" << iterCount << "... ";
			DWORD start = GetTickCount();
			newIterCount = gcapp.nextIter();
			DWORD end = GetTickCount();
			cout << "Lasting Time: " << end - start << endl;
			if (newIterCount > iterCount)
			{
				gcapp.showImage();
				cout << iterCount << ">" << endl;
			}
			else
				cout << "rect must be determined>" << endl;
			break;
		}
		case 'b':
		{
			cout << "Border Matting! ..." << endl;
			gcapp.gc.BoardMatting();
			break;
		}
		}
	}

exit_main:
	destroyWindow(winName);
	return 0;
}