#pragma once
#include "Filter.hpp"

namespace Haha
{
	class CarGameFilterImpl : public Filter
	{
	public:
		CarGameFilterImpl();
		~CarGameFilterImpl();
		void Affect(cv::Mat& img) override;

	protected:
		bool DetectionFace(cv::Mat& img, std::vector<cv::Rect>& faces);
	private:
		std::shared_ptr<cv::CascadeClassifier> _face_cascade;
		const std::string _faceCascadeFileName;
		cv::Rect _lastFaceRect;
		int _vmin;
		int _vmax;
		int _smin;
		bool _missingFaceObject;
		int _trackObject;
		cv::Mat _hue_img;
		cv::Mat _hist_img;
		cv::Mat _backproj_img;
		const std::string _backgroundFileName;
		cv::Mat _background_img;
		const std::string _carFileName;
		cv::Mat _car_img;
	};
}