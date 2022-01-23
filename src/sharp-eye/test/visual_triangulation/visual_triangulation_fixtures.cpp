#include <sharp-eye/visual_triangulation_fixtures.hpp>

VisualTriangulationTest::VisualTriangulationTest() {
    
    // Create display windows
    OPENCV_WINDOW_LEFT = "Left Image window";
    OPENCV_WINDOW_RIGHT = "Right Image window";
    cv::namedWindow(OPENCV_WINDOW_LEFT);
    cv::namedWindow(OPENCV_WINDOW_RIGHT);
    // Set image indices
    image_idx = 0;
    image_idx_max = 0;
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
    image_idx = 0;
    image_idx_max = 10;
}

void VisualTriangulationTest::TearDown(){
    cv::destroyAllWindows();
  
}