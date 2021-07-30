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
        feature.keypoint = keypoint;
        features.push_back(feature);
    }

    if(draw){
        cv::drawKeypoints(*img_ptr, keypoints, *img_ptr, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
    }

    return features;
};

FeatureVector VisualTriangulation::ExtractKeypointDescriptors(cv::Mat* img_ptr,FeatureVector &feature_vec){
    /**
     * @brief Extract the keypoint descriptor for a feature vector.
     * Returns a FeatureVector (std vector of KeypointWD) with the descriptor
     * field filled out
     */
    FeatureVector out_vec;
    if(feature_vec.empty()){
        std::cout<<"Warning : Feature vector was empty, no descriptors could be added"<<std::endl;
        return out_vec;
    }
    
    std::vector<cv::KeyPoint> keypoints;
    for(VisualSlamBase::KeypointWD &feature : feature_vec){
        cv::Mat descriptor;
        keypoints.push_back(feature.keypoint);
        orb_descriptor->compute(*img_ptr,keypoints,descriptor);
        feature.descriptor = descriptor;

        VisualSlamBase::KeypointWD out_feature;
        out_feature = feature;
        out_feature.descriptor = descriptor;
        out_vec.push_back(out_feature);
    }
    return out_vec; 
};

VisualTriangulation::~VisualTriangulation(){
    return;
};