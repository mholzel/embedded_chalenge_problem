#pragma once

#if __has_include(<filesystem>)

#include <filesystem>
namespace fs = std::filesystem;

#elif __has_include(<experimental/filesystem>)

#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

#elif __has_include(<boost/filesystem>)

#include <boost/filesystem>
namespace fs = boost::filesystem;

#else

static_assert(false, "There is no available filesystem header on this system")

#endif
