#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <fstream>


// Utiltity functions for processing file buffers and opening image files

/**
 * @brief Get the Image Filenames From Buffer object
 * 
 * @param fb 
 * @return std::vector<std::vector<std::string>> 
 */
std::vector<std::vector<std::string>> GetImageFilenamesFromBuffer(std::filebuf &fb);

/**
 * @brief Get the Image From Filename string
 * 
 * @param filename 
 * @return cv::Mat 
 */
cv::Mat GetImageFromFilename(std::string filename);