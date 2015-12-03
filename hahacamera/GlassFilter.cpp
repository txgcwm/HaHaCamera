
#include "Global.hpp"
#include "GlassFilter.hpp"

Haha::GlassFilterImpl::GlassFilterImpl()
	: Filter(FilterType::GlassFilter)
	, _faceCascadeFileName("resources/haarcascade_frontalface_alt2.xml")
	, _face_cascade(nullptr)
	, _eyeCascadeFileName("resources/haarcascade_eye_tree_eyeglasses.xml")
	, _eye_cascade(nullptr)
	, _glass_res(cv::imread("resources/glass.png"))
	, _missingObject(true)
	, _trackObject(-1)
	, _vmin(10)
	, _vmax(256)
	, _smin(30)
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
		auto faceX = faces[0].x <= 0 ? 0 : faces[0].x;
		faceX = faceX > img.cols ? img.cols : faceX;
		auto faceY = faces[0].y <= 0 ? 0 : faces[0].y;
		faceY = faceY > img.rows ? img.rows : faceY;
		auto faceWidth = faceX + faces[0].width >= img.cols ? img.cols - faceX : faces[0].width;
		auto faceHeight = faceY + faces[0].height >= img.rows ? img.rows - faceY : faces[0].height;
		auto faceRect = cv::Rect(faceX, faceY, faceWidth, faceHeight);
		std::vector<cv::Rect> eyes;
		if (faceWidth > 0 && faceHeight > 0)
		{
			cv::Mat selected = cv::Mat(img, faceRect);
			DetectionEye(selected, eyes);
		}

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

			auto glass_new_x = glass_x <= 0 ? 0 : glass_x;
			glass_new_x = glass_new_x > img.cols ? img.cols : glass_new_x;
			auto glass_new_y = glass_y <= 0 ? 0 : glass_y;
			glass_new_y = glass_new_y > img.rows ? img.rows : glass_new_y;
			auto glass_new_width = glass_new_x + glass_width >= img.cols ? img.cols - glass_new_x : glass_width;
			auto glass_new_height = glass_new_y + glass_height >= img.rows ? img.rows - glass_new_y : glass_height;
			auto glass_new_rect = cv::Rect(glass_new_x, glass_new_y, glass_new_width, glass_new_height);

			cv::Mat new_glass;
			cv::resize(_glass_res, new_glass, cv::Size(glass_new_width, glass_new_height));

			std::cout << "glass rect: " << glass_new_rect << std::endl;
			cv::Mat imageROI = img(glass_new_rect);
			cv::addWeighted(imageROI, 1.0, new_glass, 1, 0, imageROI);
		}

	}
}

bool Haha::GlassFilterImpl::DetectionFace(cv::Mat& img, std::vector<cv::Rect>& faces)
{
	if (nullptr == _face_cascade || _face_cascade->empty()) { return false; }
	if (_missingObject) //use CascadeClassifier
	{
		cv::Mat gray_img;
		cvtColor(img, gray_img, CV_BGR2GRAY);
		equalizeHist(gray_img, gray_img);
		_face_cascade->detectMultiScale(img, faces, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30), cv::Size(0, 0));
		if (faces.size() >= 1)
		{
			_missingObject = false;
			_trackObject = -1;
			_lastRect = cv::Rect(faces[0]);
		}
		return true;
	}
	else //use CamShift
	{
		cv::Mat hsv_img, mask_img, histimg_img = cv::Mat::zeros(200, 320, CV_8UC3);
		const int hsize = 16;
		float hranges[] = { 0,180 };
		const float* phranges = hranges;
		cv::cvtColor(img, hsv_img, CV_BGR2HSV); //convert image color to HSV from BGR
		cv::inRange(hsv_img, cv::Scalar(0, _smin, std::min(_vmin, _vmax)),
			cv::Scalar(180, 256, std::max(_vmin, _vmax)), mask_img);
		int ch[] = { 0, 0 };
		_hue_img.create(hsv_img.size(), hsv_img.depth());
		cv::mixChannels(&hsv_img, 1, &_hue_img, 1, ch, 1);

		if (_trackObject < 0)
		{
			cv::Mat roi(_hue_img, _lastRect), maskroi(mask_img, _lastRect);
			calcHist(&roi, 1, 0, maskroi, _hist_img, 1, &hsize, &phranges);
			cv::normalize(_hist_img, _hist_img, 0, 255, CV_MINMAX);
			_trackObject = 1;
			histimg_img = cv::Scalar::all(0);
			int binW = histimg_img.cols / hsize;
			cv::Mat buf(1, hsize, CV_8UC3);
			for (int i = 0; i < hsize; i++)
			{
				buf.at<cv::Vec3b>(i) = cv::Vec3b(cv::saturate_cast<uchar>(i*180. / hsize), 255, 255);
			}
			cv::cvtColor(buf, buf, CV_HSV2BGR);
		}

		cv::calcBackProject(&_hue_img, 1, 0, _hist_img, _backproj_img, &phranges);
		_backproj_img &= mask_img;

		cv::RotatedRect trackBox = cv::CamShift(_backproj_img, _lastRect,
			cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));
		if (trackBox.size.width < 60 || trackBox.size.height < 60)
		{
			_trackObject = -1;
			_missingObject = true;
		}
		if (_lastRect.area() <= 1)
		{
			int cols = _backproj_img.cols, rows = _backproj_img.rows, r = (std::min(cols, rows) + 5) / 6;
			_lastRect = cv::Rect(_lastRect.x - r, _lastRect.y - r,
				_lastRect.x + r, _lastRect.y + r) &
				cv::Rect(0, 0, cols, rows);
		}

		cv::ellipse(img, trackBox, cv::Scalar(0, 200, 255), 3, CV_AA);	//draw face for track object
		faces.push_back(cv::Rect(trackBox.boundingRect()));
		return true;
	}
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
