#pragma once
#include "Global.hpp"
#include "Filter.hpp"

namespace Haha
{
	class FilterManager
	{
	public:
		FilterManager();
		~FilterManager();

		bool AddFilter(FilterType type);
		bool RemoveFilter(FilterType type);
		void Affects(cv::Mat& img);
	private:
		std::list<std::shared_ptr<Haha::Filter>> _filters;
	};
}