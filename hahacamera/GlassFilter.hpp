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
	};
}