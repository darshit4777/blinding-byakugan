#include <sharp-eye/visual_triangulation.hpp>

VisualTriangulation::VisualTriangulation(const VisualSlamBase::Camera &left,const VisualSlamBase::Camera &right){
        
        // Copy the cameras over
        camera_l = left;
        camera_r = right;

        // Initializing the ORB Feature Detector
        orb_detector = cv::ORB::create();

        // Initialzing the ORB Feature Descriptor
        orb_descriptor = cv::ORB::create();
        
        //detector->detect(src,keypoints);
        //drawKeypoints(dst, keypoints, dst, Scalar::all(-1), DrawMatchesFlags::DRAW_OVER_OUTIMG);
        std::cout<<"Visual Triangulation Initialized"<<std::endl;
        
    };