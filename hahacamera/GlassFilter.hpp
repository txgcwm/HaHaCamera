#pragma once
#include "Filter.hpp"

namespace Haha
{
	class GlassFilterImpl : public Filter
	{
	public:
		GlassFilterImpl();
		~GlassFilterImpl();
		void Affect(cv::Mat& img) override;

	protected:
		bool DetectionFace(cv::Mat& img, std::vector<cv::Rect>& faces);
		bool DetectionEye(cv::Mat& img, std::vector<cv::Rect>& eyes);
	private:
		std::shared_ptr<cv::CascadeClassifier> _face_cascade;
		const std::string _faceCascadeFileName;
		std::shared_ptr<cv::CascadeClassifier> _eye_cascade;
		const std::string _eyeCascadeFileName;
		cv::Mat _glass_res;
		cv::Rect _lastRect;
		int _vmin;
		int _vmax;
		int _smin;
		bool _missingObject;
		int _trackObject;
		cv::Mat _hue_img;
		cv::Mat _hist_img;
		cv::Mat _backproj_img;
	};
}