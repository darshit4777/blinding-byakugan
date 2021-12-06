#include <sharp-eye/visual_triangulation.hpp>

typedef std::vector<KeypointWD> FeatureVector;
typedef std::vector<Framepoint> FramepointVector;
typedef std::vector<std::pair<KeypointWD,KeypointWD>> MatchVector;

VisualTriangulation::VisualTriangulation(){
        
        // Copy the cameras over
        //camera_l = left;
        //camera_r = right;

        // Initializing the ORB Feature Detector
        orb_detector = cv::ORB::create(50);
        
        // Initializing the matcher
        
        matcher = cv::FlannBasedMatcher(new cv::flann::LshIndexParams(20, 10, 2));
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
    
    //  Mask creator for binning in feature detection
    for(int i =0; i < 4; i++){
        for(int j =0; j<4; j++){
            cv::Mat mask = cv::Mat::zeros(img_ptr->size(),CV_8U);
            cv::Mat roi(mask, cv::Rect(180*i,120*j,180,120));
            roi = cv::Scalar(255);
            std::vector<cv::KeyPoint> bin_keypoints;
            cv::Mat descriptors;
            orb_detector->detectAndCompute(*img_ptr,mask,bin_keypoints,descriptors);
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
        orb_descriptor->compute(*img_ptr,keypoints,descriptor);
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

    //std::cout<<descriptor_r.rows<<" "<<descriptor_r.cols<<std::endl;
    //std::cout<<descriptor_l.rows<<" "<<descriptor_l.cols<<std::endl;
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
    const float maximum_matching_distance = 6;
    int idx_R = 0;
    //loop over all left keypoints
    for (int idx_L = 0; idx_L < left_vec.size(); idx_L++) {
        
        //stop condition
        if (idx_R == right_vec.size()){
            break;
        }

        //the right keypoints are on a lower row - skip left
        while (left_vec[idx_L].keypoint.pt.y < right_vec[idx_R].keypoint.pt.y){
            idx_L++;
            if (idx_L == right_vec.size()){
                break;
            };
        };

        //the right keypoints are on an upper row - skip right
        while (left_vec[idx_L].keypoint.pt.y > right_vec[idx_R].keypoint.pt.y){
            idx_R++;
            if (idx_R == right_vec.size()){
                break;
            }
        }
        //search bookkeeping
        int idx_RS = idx_R;
        float dist_best = maximum_matching_distance;
        int idx_best_R = 0;
        //scan epipolar line for current keypoint at idx_L
        while (left_vec[idx_L].keypoint.pt.y == right_vec[idx_RS].keypoint.pt.y){
            //zero disparity stop condition
            if (right_vec[idx_RS].keypoint.pt.x >= left_vec[idx_L].keypoint.pt.x){
                break;
            }
            //compute descriptor distance using hamming norm
            const float dist = cv::norm(left_vec[idx_L].descriptor, right_vec[idx_RS].descriptor,cv::NORM_HAMMING);
            if(dist < dist_best){
                dist_best = dist;
                idx_best_R = idx_RS;
            };
            idx_RS++;
        };
        //check if something was found
        if (dist_best < maximum_matching_distance) {
            std::pair<KeypointWD,KeypointWD> matched_pair;
            matched_pair.first = left_vec[idx_L];
            matched_pair.second = right_vec[idx_best_R];
            
            matched_keypoints.push_back(matched_pair);
        };
    };
    return matched_keypoints;
};



VisualTriangulation::~VisualTriangulation(){
    return;
};
