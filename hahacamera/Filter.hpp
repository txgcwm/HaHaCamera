#pragma once
#include "Global.hpp"

namespace Haha
{
	enum class FilterType
	{
		GlassFilter,
		CarGameFilter,
	};
	class Filter
	{
	public:
		Filter(FilterType type) { _type = type; }
		virtual void Affect(cv::Mat& img) {}
		FilterType GetType() { return _type; }
	protected:
		FilterType _type;
	};
}