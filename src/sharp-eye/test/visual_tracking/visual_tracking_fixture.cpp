#include <sharp-eye/visual_tracking_fixture.hpp>


VisualTrackingTest::VisualTrackingTest(){
    /**
     * @brief Construct the cameras and assign the local map
     * and tracker
     * 
     */
    // Create display windows
    OPENCV_WINDOW_LEFT = "Left Image window";
    OPENCV_WINDOW_RIGHT = "Right Image window";
    cv::namedWindow(OPENCV_WINDOW_LEFT);
    cv::namedWindow(OPENCV_WINDOW_RIGHT);
    // Set image file paths
    image_path_left ="/home/darshit/Code/blinding-byakugan/MH_01_easy/MH_01_easy.txt.d/left_camera/";
    image_path_right = "/home/darshit/Code/blinding-byakugan/MH_01_easy/MH_01_easy.txt.d/right_camera/";

    std::string image_list_left_file = "/home/darshit/Code/blinding-byakugan/MH_01_easy/MH_01_easy.txt.d/left_cam_fnames.txt"; 
    std::string image_list_right_file = "/home/darshit/Code/blinding-byakugan/MH_01_easy/MH_01_easy.txt.d/right_cam_fnames.txt";

    std::filebuf image_list_left_buf;
    std::filebuf image_list_right_buf;

    image_list_left_buf.open(image_list_left_file,std::ios::in);
    image_list_right_buf.open(image_list_right_file,std::ios::in);

    cam_left_image_list = GetImageFilenamesFromBuffer(image_list_left_buf);
    cam_right_image_list = GetImageFilenamesFromBuffer(image_list_right_buf);

    cam_left.intrinsics << 435.20,     0.0,    367.215,
                               0.0, 435.20,    252.375,
                               0.0,     0.0,        1.0;

    cam_right.intrinsics << 435.20,        0.0, 367.215,
                                0.0,    435.134, 252.375,
                                0.0,        0.0,    1.0;                                

    
    tracker = new VisualTracking(cam_left,cam_right);
    focal_length = 435.20;
    baseline = 0.110074;
}

VisualTrackingTest::~VisualTrackingTest(){
    delete tracker;
}

void VisualTrackingTest::SetUp(){
    return;
}

void VisualTrackingTest::TearDown(){
    return;
}

boost::shared_ptr<Frame> VisualTrackingTest::GetFrame(cv::Mat image_l, cv::Mat image_r){
    /**
     * @brief Performs keypoint extraction, epipolar matching and 
     * triangulation. The framepoints so created are then packed
     * into a frame object.
     * 
     */

    // Feature Detection
    features_l.clear();
    features_l = triangulator.DetectAndComputeFeatures(&image_l,features_l,false);

    features_r.clear();
    features_r = triangulator.DetectAndComputeFeatures(&image_r,features_r,false);

    // Generate epipolar matching
    matches = triangulator.GetEpipolarMatches(features_l,features_r);

    // Triangulation
    FramepointVector framepoints;
    triangulator.Generate3DCoordinates(matches,framepoints,baseline,focal_length,cam_left.intrinsics);

    // Frame Generation
    Frame frame;
    frame.camera_l = cam_left;
    frame.camera_r = cam_right;

    frame.image_l = image_l;
    frame.image_r = image_r;
    
    // Generate a vector of framepoint ptrs
    FramepointPointerVector fp_ptr_vec;
    for(auto framepoint : framepoints){
        // Create a heap declaration of the framepoint and add a shared pointer
        boost::shared_ptr<Framepoint> fp_ptr = boost::make_shared<Framepoint>(framepoint);
        fp_ptr_vec.push_back(fp_ptr);
    }
    frame.points = fp_ptr_vec;
    frame.T_cam2world.setIdentity();
    frame.T_world2cam.setIdentity();

    boost::shared_ptr<Frame> frame_ptr = boost::make_shared<Frame>(frame);
    return frame_ptr;

}

cv::Mat VisualTrackingTest::GetImage(int image_idx,std::string cam){
    std::string image_filename;
    
    try{
        if (cam=="left"){
            image_filename = image_path_left + cam_left_image_list[image_idx][0];
        }
        if(cam=="right"){
            image_filename = image_path_right + cam_right_image_list[image_idx][0];
        }
        return GetImageFromFilename(image_filename);
    }
    catch(std::exception& e){
        std::cout<<e.what()<<std::endl;
    };
}