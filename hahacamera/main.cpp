#include "Global.hpp"
#include "FilterManager.hpp"
#include "GlassFilter.hpp"

int main(int argc, char *args[])
{
	Haha::FilterManager filterMgr;
	//filterMgr.AddFilter(Haha::FilterType::GlassFilter);
	filterMgr.AddFilter(Haha::FilterType::CarGameFilter);
	cv::Mat frame;
	cv::VideoCapture camera(0);
	if (!camera.isOpened())
	{
		std::cout << "***Could not initialize capturing...***\n";
		return 0;
	}
	while (true)
	{
		camera >> frame;
		if (frame.empty())
		{
			break;
		}
		//do filters
		filterMgr.Affects(frame);

		cv::imshow("HaHaCamera", frame);
		cv::waitKey(10);
	}
	
	return 0;
}
