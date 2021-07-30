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

MatchVector VisualTriangulation::GetKeypointMatches(FeatureVector &left_vec, FeatureVector &right_vec){
    /**
     * @brief Uses the FLANN matcher to compute matches between two feature vectors
     * 
     */

    // Preprocessing 
    // Before calling the matcher, we need to arrange our datastructures
    cv::Mat descriptor_l, descriptor_r;
    std::vector<cv::KeyPoint> keypoint_vec_l, keypoint_vec_r;

    for(int i =0; i < left_vec.size(); i++){
        descriptor_l.row(i) = left_vec[i].descriptor;
        keypoint_vec_l[i] = left_vec[i].keypoint;
    }

    for(int i =0; i < right_vec.size(); i++){
        descriptor_r.row(i) = right_vec[i].descriptor;
        keypoint_vec_r[i] = right_vec[i].keypoint;
    }

    // Calling the matcher to create matches
    // The nomenclature used for matching is query and train. 
    // The query set is the input set for which you want to find matches
    // The train set is the set in which you want to find matches to the query set,

    std::vector< std::vector<cv::DMatch> > knn_matches;
    
    matcher.knnMatch(descriptor_l,descriptor_r,knn_matches,2);

    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    };

    // Now we have a vector with good matches holding the query indices
    // and the train indices

    MatchVector matched_vector;
    for(int i = 0; i < good_matches.size(); i++){
        VisualSlamBase::KeypointWD keypoint_l, keypoint_r;
        std::pair<VisualSlamBase::KeypointWD,VisualSlamBase::KeypointWD> matched_pair;
        keypoint_l = left_vec[i];
        keypoint_r = right_vec[i];

        matched_pair.first = keypoint_l;
        matched_pair.second = keypoint_r;

        matched_vector.push_back(matched_pair);
    }

    return matched_vector;
};

VisualTriangulation::~VisualTriangulation(){
    return;
};
