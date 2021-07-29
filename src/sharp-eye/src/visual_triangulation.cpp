#include <sharp-eye/visual_triangulation.hpp>

typedef std::vector<VisualSlamBase::KeypointWD> FeatureVector;
typedef std::vector<std::pair<VisualSlamBase::KeypointWD,VisualSlamBase::KeypointWD>> MatchVector;

VisualTriangulation::VisualTriangulation(){
        
        // Copy the cameras over
        //camera_l = left;
        //camera_r = right;

        // Initializing the ORB Feature Detector
        orb_detector = cv::ORB::create();

        // Initialzing the ORB Feature Descriptor
        orb_descriptor = cv::ORB::create();
        
        //detector->detect(src,keypoints);
        //drawKeypoints(dst, keypoints, dst, Scalar::all(-1), DrawMatchesFlags::DRAW_OVER_OUTIMG);
        std::cout<<"Visual Triangulation Initialized"<<std::endl;
        
    };

FeatureVector VisualTriangulation::DetectFeatures(cv::Mat* img_ptr,bool draw){
    /**
     * @brief Perform feature detectior on the input image using ORB detector
     * The words features and keypoints are used interchangeably and are introduced
     * to avoid variable confusion. Features will always refer to keypoint with
     * descriptor
     */
    std::vector<cv::KeyPoint> keypoints;
    orb_detector->detect(*img_ptr,keypoints);
    FeatureVector features;

    if(keypoints.empty()){
        std::cout<<"Warning : No keypoints detected in image"<<std::endl;
        return features;
    };
    // Copy over the keypoints vector into the feature vector
    for(cv::KeyPoint keypoint : keypoints){
        VisualSlamBase::KeypointWD feature;
        feature.pixel_x = keypoint.pt.x;
        feature.pixel_y = keypoint.pt.y;
        feature.response = keypoint.response;
        features.push_back(feature);
    }

    if(draw){
        cv::drawKeypoints(*img_ptr, keypoints, *img_ptr, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
    }

    return features;
};

VisualTriangulation::~VisualTriangulation(){
    return;
};