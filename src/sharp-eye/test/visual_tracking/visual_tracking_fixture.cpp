#include <sharp-eye/visual_tracking_fixture.hpp>


VisualTrackingTest::VisualTrackingTest(){
    /**
     * @brief Construct the cameras and assign the local map
     * and tracker
     * 
     */

    cam_left.intrinsics << 458.654,     0.0,    367.215,
                               0.0, 457.296,    248.375,
                               0.0,     0.0,        1.0;

    cam_right.intrinsics << 457.587,        0.0, 379.999,
                                0.0,    456.134, 255.238,
                                0.05,        0.0,    1.0;                                

    
    lmap_ptr = new LocalMap;
    tracker = new VisualTracking(cam_left,cam_right);
    focal_length = 457.975;
    baseline = 0.11;
}

VisualTrackingTest::~VisualTrackingTest(){
    delete lmap_ptr;
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