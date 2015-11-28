#include "Global.hpp"
#include "FilterManager.hpp"
#include "GlassFilter.hpp"

int main(int argc, char *args[])
{
	Haha::FilterManager filterMgr;
	filterMgr.AddFilter(Haha::FilterType::GlassFilter);
	cv::Mat frame;
	cv::VideoCapture camera(0);
	while (true)
	{
		if (!camera.isOpened())
		{
			cv::waitKey(10); continue;
		}
		camera >> frame;
		//do filters
		filterMgr.Affects(frame);

		cv::imshow("HaHaCamera", frame);
		cv::waitKey(10);
	}
	
	return 0;
}
