#include "Global.hpp"
#include "FilterManager.hpp"
#include "GlassFilter.hpp"

Haha::FilterManager::FilterManager()
{
}

Haha::FilterManager::~FilterManager()
{
}

bool Haha::FilterManager::AddFilter(FilterType type)
{
	if (type == FilterType::GlassFilter)
	{
		_filters.push_back(std::make_shared<Haha::GlassFilterImpl>());
	}
	return true;
}

bool Haha::FilterManager::RemoveFilter(FilterType type)
{
	_filters.remove_if(
		[&type](auto val) {
		return val->GetType() == type;
	});
	return false;
}

void Haha::FilterManager::Affects(cv::Mat & img)
{
	for (auto& filter : _filters)
	{
		filter->Affect(img);
	}
}
