#include "Global.hpp"
#include "CarGameFilter.hpp"


Haha::CarGameFilterImpl::CarGameFilterImpl()
	: Filter(FilterType::CarGameFilter)
	, _faceCascadeFileName("resources/haarcascade_frontalface_alt2.xml")
	, _face_cascade(nullptr)
	, _missingFaceObject(true)
	, _trackObject(-1)
	, _vmin(20)
	, _vmax(256)
	, _smin(30)
{
	_face_cascade = std::make_shared<cv::CascadeClassifier>();
	auto ret = _face_cascade->load(_faceCascadeFileName);
	if (!ret)
	{
		std::cout << "ERROR: Could not load classifier cascade by CPU: " << _faceCascadeFileName << std::endl;
	}
}

Haha::CarGameFilterImpl::~CarGameFilterImpl()
{
}

void Haha::CarGameFilterImpl::Affect(cv::Mat& img)
{
	std::vector<cv::Rect> faces;
	DetectionFace(img, faces);

	if (faces.size() >= 1)
	{
		auto faceX = faces[0].x <= 0 ? 0 : faces[0].x;
		faceX = faceX > img.cols ? img.cols : faceX;
		auto faceY = faces[0].y <= 0 ? 0 : faces[0].y;
		faceY = faceY > img.rows ? img.rows : faceY;
		auto faceWidth = faceX + faces[0].width >= img.cols ? img.cols - faceX : faces[0].width;
		auto faceHeight = faceY + faces[0].height >= img.rows ? img.rows - faceY : faces[0].height;
		auto faceRect = cv::Rect(faceX, faceY, faceWidth, faceHeight);

		if (faceWidth > 0 && faceHeight > 0)
		{
			cv::rectangle(img, faceRect, cv::Scalar(0, 255, 0), 3);
		}
	}
}

bool Haha::CarGameFilterImpl::DetectionFace(cv::Mat& img, std::vector<cv::Rect>& faces)
{
	if (nullptr == _face_cascade || _face_cascade->empty()) { return false; }
	if (_missingFaceObject) //use CascadeClassifier
	{
		cv::Mat gray_img;
		cvtColor(img, gray_img, CV_BGR2GRAY);
		equalizeHist(gray_img, gray_img);
		_face_cascade->detectMultiScale(img, faces, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30), cv::Size(0, 0));
		if (faces.size() >= 1)
		{
			_missingFaceObject = false;
			_trackObject = -1;
			_lastFaceRect = cv::Rect(faces[0]);
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
			cv::Mat roi(_hue_img, _lastFaceRect), maskroi(mask_img, _lastFaceRect);
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

		cv::RotatedRect trackBox = cv::CamShift(_backproj_img, _lastFaceRect,
			cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));
		if (trackBox.size.width < 60 || trackBox.size.height < 60)
		{
			_trackObject = -1;
			_missingFaceObject = true;
		}
		if (_lastFaceRect.area() <= 1)
		{
			int cols = _backproj_img.cols, rows = _backproj_img.rows, r = (std::min(cols, rows) + 5) / 6;
			_lastFaceRect = cv::Rect(_lastFaceRect.x - r, _lastFaceRect.y - r,
				_lastFaceRect.x + r, _lastFaceRect.y + r) &
				cv::Rect(0, 0, cols, rows);
		}
		faces.push_back(cv::Rect(trackBox.boundingRect()));
		return true;
	}
}
