#include <sharp-eye/utils.hpp>


std::vector<std::vector<std::string>> GetImageFilenamesFromBuffer(std::filebuf &fb){
    // Reads the image filenames from a csv - uses opencv to load the images and 
    // assigns the images to image_l and image_r

    std::istream file(&fb);
    
    std::vector<std::vector<std::string>> row;
    std::vector<std::string>   result;
    std::string                line;
    while(!file.eof()){
        
        // Get a new line
        std::getline(file,line);

        // Add the line to a string stream
        std::stringstream          lineStream(line);
        std::string                cell;

        while(std::getline(lineStream,cell, ','))
        {
            result.push_back(cell);
        }

        // This checks for a trailing comma with no data after it.
        if (!lineStream && cell.empty())
        {
            // If there was a trailing comma then add an empty element.
            result.push_back("");
        }
        row.push_back(result);
        result.clear();
    }
    return row;   
}

cv::Mat GetImageFromFilename(std::string filename){
    return cv::imread(filename);
}