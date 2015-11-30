
#include "Global.hpp"
#include "GlassFilter.hpp"

Haha::GlassFilterImpl::GlassFilterImpl()
	: Filter(FilterType::GlassFilter)
	,_faceCascadeFileName("resources/haarcascade_frontalface_alt2.xml")
	, _face_cascade(nullptr)
	, _eyeCascadeFileName("resources/haarcascade_eye_tree_eyeglasses.xml")
	, _eye_cascade(nullptr)
	, _glass_res(cv::imread("resources/glass.png"))
{
	_face_cascade = std::make_shared<cv::CascadeClassifier>();
	auto ret = _face_cascade->load(_faceCascadeFileName);
	if (!ret)
	{
		std::cout << "ERROR: Could not load classifier cascade by CPU: " << _faceCascadeFileName << std::endl;
	}
	_eye_cascade = std::make_shared<cv::CascadeClassifier>();
	ret = _eye_cascade->load(_eyeCascadeFileName);
	if (!ret)
	{
		std::cout << "ERROR: Could not load classifier cascade by CPU: " << _eyeCascadeFileName << std::endl;
	}
}

Haha::GlassFilterImpl::~GlassFilterImpl()
{
}

void Haha::GlassFilterImpl::Affect(cv::Mat& img)
{
	std::vector<cv::Rect> faces;
	DetectionFace(img, faces);

	if (faces.size() >= 1) {
		std::vector<cv::Rect> eyes;
		DetectionEye(img(faces[0]), eyes);

		if (eyes.size() >= 2) {
			auto eye_center1 = cv::Point2f(faces[0].x + eyes[0].x + static_cast<float>( eyes[0].width ) / 2, faces[0].y + eyes[0].y + static_cast<float>( eyes[0].height ) / 2);
			auto eye_center2 = cv::Point2f(faces[0].x + eyes[1].x + eyes[1].width / 2, faces[0].y + eyes[1].y + eyes[1].height / 2);
			auto center_point = cv::Point2f((eye_center1.x + eye_center2.x) * 0.5f, (eye_center1.y + eye_center2.y) * 0.5f);
			auto eyes_distance = std::sqrt(std::pow(eye_center1.x - eye_center2.x, 2)
								 + std::pow(eye_center1.y - eye_center2.y, 2));
			auto glass_width = eyes_distance / 0.566f;
			auto glass_height = glass_width * _glass_res.rows / _glass_res.cols;
			auto glass_x = center_point.x - glass_width * 0.5f;
			auto glass_y = center_point.y - glass_height * 0.5f;

			cv::Mat new_glass;
			cv::resize(_glass_res, new_glass, cv::Size(glass_width, glass_height));

			std::cout << "eyes_distance: " << eyes_distance << ", glass_width: " << glass_width << ", glass_height: " << glass_height << std::endl;
			std::cout << "glass_x " << glass_x << ", glass_y: " << glass_x << ", calc width: " << (glass_x + glass_width > img.cols ? img.cols - glass_x : glass_width) 
				<< ", calc height: " << (glass_y + glass_height > img.rows ? img.rows - glass_y : glass_height) << std::endl;
			cv::Mat imageROI = img(cv::Rect(
									glass_x < 0 ? 0 : glass_x, glass_y < 0 ? 0 : glass_y,
									glass_x + glass_width > img.cols ? img.cols - glass_x : glass_width,
									glass_y + glass_height > img.rows ? img.rows - glass_y : glass_height));
			cv::addWeighted(imageROI, 1.0, new_glass, 1, 0, imageROI);
		}

	}
}

bool Haha::GlassFilterImpl::DetectionFace(cv::Mat& img, std::vector<cv::Rect>& faces)
{
	if (nullptr == _face_cascade || _face_cascade->empty()) { return false; }
	cv::Mat gray_img;
	cvtColor(img, gray_img, CV_BGR2GRAY);
	equalizeHist(gray_img, gray_img);
	_face_cascade->detectMultiScale(img, faces, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30), cv::Size(0, 0));
	return true;
}

bool Haha::GlassFilterImpl::DetectionEye(cv::Mat& img, std::vector<cv::Rect>& eyes)
{
	if (nullptr == _eye_cascade || _eye_cascade->empty()) { return false; }
	cv::Mat gray_img;
	cvtColor(img, gray_img, CV_BGR2GRAY);
	equalizeHist(gray_img, gray_img);
	_eye_cascade->detectMultiScale(img, eyes, 1.1, 3.0);
	return false;
}
