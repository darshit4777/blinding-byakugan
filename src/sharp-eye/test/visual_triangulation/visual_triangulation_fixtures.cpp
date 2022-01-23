#include <sharp-eye/visual_triangulation_fixtures.hpp>
#include <sharp-eye/utils.hpp>

VisualTriangulationTest::VisualTriangulationTest() {
    
    // Create display windows
    OPENCV_WINDOW_LEFT = "Left Image window";
    OPENCV_WINDOW_RIGHT = "Right Image window";
    cv::namedWindow(OPENCV_WINDOW_LEFT);
    cv::namedWindow(OPENCV_WINDOW_RIGHT);
    // Set image file paths
    image_path_left ="/home/darshit/Code/blinding-byakugan/MH_01_easy/MH_01_easy.txt.d/left_camera/";
    image_path_right = "/home/darshit/Code/blinding-byakugan/MH_01_easy/MH_01_easy.txt.d/right_camera/";
    return;
}

VisualTriangulationTest::~VisualTriangulationTest(){
 // You can do clean-up work that doesn't throw exceptions here.
}

// If the constructor and destructor are not enough for setting up
// and cleaning up each test, you can define the following methods:

void VisualTriangulationTest::SetUp() {
    std::string image_list_left_file = "/home/darshit/Code/blinding-byakugan/MH_01_easy/MH_01_easy.txt.d/left_cam_fnames.txt"; 
    std::string image_list_right_file = "/home/darshit/Code/blinding-byakugan/MH_01_easy/MH_01_easy.txt.d/right_cam_fnames.txt";

    std::filebuf image_list_left_buf;
    std::filebuf image_list_right_buf;

    image_list_left_buf.open(image_list_left_file,std::ios::in);
    image_list_right_buf.open(image_list_right_file,std::ios::in);

    cam_left_image_list = GetImageFilenamesFromBuffer(image_list_left_buf);
    cam_right_image_list = GetImageFilenamesFromBuffer(image_list_right_buf);
}

cv::Mat VisualTriangulationTest::GetImage(int image_idx,std::string cam){
    std::string image_filename;
    
    try{
        if (cam=="left"){
            image_filename = image_path_left + cam_left_image_list[image_idx][0];
        }
        if(cam=="right"){
            image_filename = image_path_left + cam_left_image_list[image_idx][0];
        }
        return GetImageFromFilename(image_filename);
    }
    catch(std::exception& e){
        std::cout<<e.what()<<std::endl;
    };
}

void VisualTriangulationTest::TearDown(){
    cv::destroyAllWindows(); 
}