#include <sharp-eye/visual_triangulation.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
typedef std::vector<KeypointWD> FeatureVector;
typedef std::vector<Framepoint> FramepointVector;
typedef std::vector<std::pair<KeypointWD,KeypointWD>> MatchVector;

VisualTriangulation::VisualTriangulation(){
        
        // Initializing the ORB Feature Detector
        orb_detector = cv::ORB::create();
        fast_feature_threshold = 20;
        fast_detector = cv::FastFeatureDetector::create(fast_feature_threshold);
        feature_descriptor = cv::ORB::create();
        
        horizontal_bins = 1;
        vertical_bins = 1;
        min_keypoints_in_bin = 1;
        
        // Initializing the matcher
        matcher = cv::FlannBasedMatcher(new cv::flann::LshIndexParams(20, 10, 2));
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
        KeypointWD feature;
        feature.keypoint = keypoint;
        features.push_back(feature);
    }

    if(draw){
        cv::drawKeypoints(*img_ptr, keypoints, *img_ptr, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
    }

    return features;
};

FeatureVector VisualTriangulation::DetectAndComputeFeatures(cv::Mat* img_ptr,FeatureVector &features,bool draw){
    
    std::vector<cv::KeyPoint> image_keypoints;
    int img_width = img_ptr->cols;
    int img_height = img_ptr->rows;
    //  Mask creator for binning in feature detection
    for(int i =0; i < horizontal_bins; i++){
        for(int j =0; j<vertical_bins; j++){
            cv::Mat mask = cv::Mat::zeros(img_ptr->size(),CV_8U);        
            cv::Mat roi(mask, cv::Rect(i*img_width/horizontal_bins,j*img_height/vertical_bins,img_width/horizontal_bins,img_height/vertical_bins));
            roi = cv::Scalar(255);
            
            std::vector<cv::KeyPoint> bin_keypoints;
            cv::Mat descriptors;
            fast_detector->detect(*img_ptr,bin_keypoints,mask);
            if(bin_keypoints.size() < min_keypoints_in_bin){
                // Provision to re-run the feature detector if a particular bin has less than minimum features.
                fast_detector->setThreshold(fast_feature_threshold/2);
                bin_keypoints.clear();
                fast_detector->detect(*img_ptr,bin_keypoints,mask);
                fast_detector->setThreshold(fast_feature_threshold);
            };
            
            feature_descriptor->compute(*img_ptr,bin_keypoints,descriptors);


            for(int i = 0; i < bin_keypoints.size(); i++){
                KeypointWD feature;
                image_keypoints.push_back(bin_keypoints[i]);
                feature.keypoint = bin_keypoints[i];
                feature.descriptor = cv::Mat(descriptors.row(i));
                features.push_back(feature);
            }
        }
    }

    if(features.empty()){
        std::cout<<"Warning : No keypoints detected in image"<<std::endl;
        return features;
    };
    

    if(draw){
        cv::drawKeypoints(*img_ptr, image_keypoints, *img_ptr, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
    }

    return features;
}

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
    for(KeypointWD &feature : feature_vec){
        cv::Mat descriptor;
        keypoints.push_back(feature.keypoint);
        feature_descriptor->compute(*img_ptr,keypoints,descriptor);
        
        feature.descriptor = descriptor;

        KeypointWD out_feature;
        out_feature = feature;
        out_feature.descriptor = descriptor;
        out_vec.push_back(out_feature);
    }
    return out_vec; 
};

MatchVector VisualTriangulation::GetKeypointMatches(FeatureVector &left_vec, FeatureVector &right_vec){
    /**
     * @brief Uses the KNN matcher to compute matches between two feature vectors
     * 
     */

    MatchVector matched_vector;
    // Preprocessing 
    // Before calling the matcher, we need to arrange our datastructures
    cv::Mat descriptor_l;
    cv::Mat descriptor_r;
    
    std::vector<cv::KeyPoint> keypoint_vec_l, keypoint_vec_r;

    for(int i =0; i < left_vec.size(); i++){
        descriptor_l.push_back(left_vec[i].descriptor);
        keypoint_vec_l.push_back(left_vec[i].keypoint);
    }

    for(int i =0; i < right_vec.size(); i++){
        descriptor_r.push_back(right_vec[i].descriptor);
        keypoint_vec_r.push_back(right_vec[i].keypoint);
    }

    // Calling the matcher to create matches
    // The nomenclature used for matching is query and train. 
    // The query set is the input set for which you want to find matches
    // The train set is the set in which you want to find matches to the query set,
    
    std::vector< std::vector<cv::DMatch> > knn_matches;
    if(descriptor_r.empty()){
        std::cout<<"Warning - No Descriptors for Right camera"<<std::endl;
        return matched_vector;
    }
    if(descriptor_l.empty()){
        std::cout<<"Warning - No Descriptors for Left camera"<<std::endl;
        return matched_vector;
    };
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

    
    for(int i = 0; i < good_matches.size(); i++){
        KeypointWD keypoint_l, keypoint_r;
        std::pair<KeypointWD,KeypointWD> matched_pair;
        
        keypoint_l = left_vec[good_matches[i].queryIdx];
        keypoint_r = right_vec[good_matches[i].trainIdx];

        matched_pair.first = keypoint_l;
        matched_pair.second = keypoint_r;
        matched_vector.push_back(matched_pair);
    }

    if(matched_vector.empty()){
        std::cout<<"Warning : No matches between left and right images"<<std::endl;
    }
    return matched_vector;
};

FramepointVector VisualTriangulation::Generate3DCoordinates(MatchVector &matched_features,FramepointVector &framepoints_in, float baseline, float focal_length, Eigen::Matrix3f camera_intrinsics){
    /**
     * @brief Uses the camera baseline and matched keypoints from left and right
     * cameras to generate 3D coordinates of keypoints.
     * 
     */

    // Calculating the Depth and X Y points for each feature. 

    // We will consider the left camera as the origin for these calculations
    for(int i =0; i < matched_features.size(); i++){
        // Each matched feature is stored as a pair of KeypointWD
        // The first is left and the second is right
        Framepoint framepoint;
        framepoint.keypoint_l = matched_features[i].first;
        framepoint.keypoint_r = matched_features[i].second;

        Eigen::Vector3f camera_coordinates; // Coordinates in the camera frame
        
        // Calculating pixel euclidean distance
        float px_distance,xl,yl,xr,yr;
        xl = matched_features[i].first.keypoint.pt.x;
        yl = matched_features[i].first.keypoint.pt.y;
        xr = matched_features[i].second.keypoint.pt.x;
        yr = matched_features[i].second.keypoint.pt.y;
        px_distance = fabs(xl-xr);
        if(px_distance == 0){
            // Could be a point at infinity
            continue;
        };
        camera_coordinates.z() = focal_length * baseline/px_distance;
        float cx = camera_intrinsics.coeff(0,2);
        float cy = camera_intrinsics.coeff(1,2);
        float fx = camera_intrinsics.coeff(0,0);
        float fy = camera_intrinsics.coeff(1,1);
        
        camera_coordinates.x() = (xl - cx)*camera_coordinates.z()/fx;
        camera_coordinates.y() = (yl - cy)*camera_coordinates.z()/fy;

        // Now the camera coordinates are assigned to the specific framepoint
        framepoint.camera_coordinates = camera_coordinates;
        framepoints_in.push_back(framepoint);
    };

    return framepoints_in;
};

MatchVector VisualTriangulation::GetEpipolarMatches(FeatureVector &left_vec, FeatureVector &right_vec){

    MatchVector matched_keypoints;
    matched_keypoints.clear();
    
    // Large sorting expression that is explained in the ProSLAM paper

    // Sort Left
    std::sort(left_vec.begin(),left_vec.end(),[](const KeypointWD& a,const KeypointWD& b){
        return ((a.keypoint.pt.y < b.keypoint.pt.y)||(a.keypoint.pt.y == b.keypoint.pt.y && a.keypoint.pt.x < b.keypoint.pt.x));
    });

    // Sort Right
    std::sort(right_vec.begin(),right_vec.end(),[](const KeypointWD& a,const KeypointWD& b){
        return ((a.keypoint.pt.y < b.keypoint.pt.y)||(a.keypoint.pt.y == b.keypoint.pt.y && a.keypoint.pt.x < b.keypoint.pt.x));
    });

    //configuration
    const float maximum_matching_distance = 25.6;
    int idx_R = 0;
    int epipolar_tolerance = 2;
    //loop over all left keypoints
    for (int idx_L = 0; idx_L < left_vec.size(); idx_L++) {
        int epipolar_lower_line = left_vec[idx_L].keypoint.pt.y + epipolar_tolerance;
        int epipolar_upper_line = left_vec[idx_L].keypoint.pt.y - epipolar_tolerance;
        float dist = maximum_matching_distance;
        float dist_best = maximum_matching_distance;
        std::pair<KeypointWD,KeypointWD> matched_pair;
        for(auto point : right_vec){

            if(point.keypoint.pt.y == left_vec[idx_L].keypoint.pt.y){
                // This point is a candidate for matching
                // compute descriptor distance using hamming norm
                if(point.keypoint.pt.x > left_vec[idx_L].keypoint.pt.x){
                    continue;
                }
                dist = cv::norm(left_vec[idx_L].descriptor, point.descriptor,cv::NORM_HAMMING);
                if(dist < dist_best){
                    dist_best = dist;
                    matched_pair.first = left_vec[idx_L];
                    matched_pair.second = point;
                };
            };
        };
        if(dist_best < maximum_matching_distance){
            matched_keypoints.push_back(matched_pair);
        }
    }
    return matched_keypoints;
};



VisualTriangulation::~VisualTriangulation(){
    return;
};
