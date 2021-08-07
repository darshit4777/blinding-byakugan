#include<sharp-eye/visual_tracking.hpp>


typedef std::vector<VisualSlamBase::KeypointWD> FeatureVector;
typedef std::vector<VisualSlamBase::Framepoint> FramepointVector;
typedef std::vector<VisualSlamBase::Frame> FrameVector;
typedef std::vector<std::pair<VisualSlamBase::KeypointWD,VisualSlamBase::KeypointWD>> MatchVector;
typedef VisualSlamBase::Camera Camera;

VisualTracking::VisualTracking(Camera &cam_left,Camera &cam_right){

    // Initialize camera specifics
    camera_left = cam_left;
    camera_right = cam_right;
    img_height = 480;
    img_width = 720;
    
    frames.clear();
    
    matcher = cv::FlannBasedMatcher(new cv::flann::LshIndexParams(20, 10, 2));

    std::cout<<"Visual Tracking Initialized"<<std::endl;  
};

FramepointVector VisualTracking::FindCorrespondences(FramepointVector &previous_frame,FramepointVector &current_frame){
    /**
     * @brief Finds the correspondences / matches between the framepoint vector of
     * the previous frame and that of the current frame. 
     * We use the knn matcher applied to small rectangular ROIs around each query
     * framepoint.
     * 
     * For finding correspondences, we will use the keypoints from the left image
     * only.
     */

    // Sorting both vectors
    // TODO : Check if sorting is needed, Framepoints may already have their keypoints sorted.
    
    // Sort Previous
    std::sort(previous_frame.begin(),previous_frame.end(),[](const VisualSlamBase::Framepoint& a,const VisualSlamBase::Framepoint& b){
        return ((a.keypoint_l.keypoint.pt.y < b.keypoint_l.keypoint.pt.y)||
        (a.keypoint_l.keypoint.pt.y == b.keypoint_l.keypoint.pt.y && a.keypoint_l.keypoint.pt.x < b.keypoint_l.keypoint.pt.x));
    });

    // Sort Current
    std::sort(current_frame.begin(),current_frame.end(),[](const VisualSlamBase::Framepoint& a,const VisualSlamBase::Framepoint& b){
        return ((a.keypoint_l.keypoint.pt.y < b.keypoint_l.keypoint.pt.y)||
        (a.keypoint_l.keypoint.pt.y == b.keypoint_l.keypoint.pt.y && a.keypoint_l.keypoint.pt.x < b.keypoint_l.keypoint.pt.x));
    });
    // We use the previous frame as the query frame

    for(VisualSlamBase::Framepoint &query_framepoint : previous_frame){
        // Break Condition
        int id_current = 0;
        std::vector<int> match_shortlist; // Framepoints found in the rectangular search region
        int ymin,ymax,xmin,xmax;
        ymin = std::max(int(query_framepoint.keypoint_l.keypoint.pt.y - 25),0);
        ymax = std::min(int(query_framepoint.keypoint_l.keypoint.pt.y + 25),img_height);

        xmin = std::max(int(query_framepoint.keypoint_l.keypoint.pt.x - 25),0);
        xmax = std::min(int(query_framepoint.keypoint_l.keypoint.pt.x + 25),img_width);
        
        // Loop to search for the top of the rectangular region
        while(current_frame[id_current].keypoint_l.keypoint.pt.y < ymin){
            id_current++;
        };

        // The search point is now within the rows of the rectangular region
        // We check each keypoint and see if it obeys the column constraints
        // when the lower row of the rectangular region is breached, we move 
        // to the next point
        while(current_frame[id_current].keypoint_l.keypoint.pt.y < ymax){
            if(id_current >= current_frame.size()){
                break;
            }
            int x = current_frame[id_current].keypoint_l.keypoint.pt.x;
            
            // Check if the keypoint is within the rectangle
            if((x < xmax) && (x>xmin)){
                // Within the rectangle
                match_shortlist.push_back(id_current);
            }
            id_current++;
        };

        if(match_shortlist.empty()){
            continue;
        };

        // Now that the match shortlist is created, we check for the best match
        std::vector< std::vector<cv::DMatch> > knn_matches;

        // Before calling the matcher, we need to arrange our datastructures
        cv::Mat query_descriptor;
        cv::Mat descriptor_current;
    
        // The match shortlist is a vector of shortlisted indices from current frame

        // Assigning the current and query descriptors
        for(int i =0; i < match_shortlist.size(); i++){
            //std::cout<<match_shortlist[i]<<std::endl;
            //std::cout<<current_frame.size()<<std::endl;
            descriptor_current.push_back(current_frame[match_shortlist[i]].keypoint_l.descriptor);
        };

        query_descriptor = query_framepoint.keypoint_l.descriptor;

        matcher.knnMatch(query_descriptor,descriptor_current,knn_matches,2);
        if(knn_matches[0].empty()){
            // If no matches are returned
            continue;
        };
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
        if(good_matches.empty()){

            continue;
        }
        // Good matches consists of the id of the query descriptor which will always be zero
        // and the index of the descriptor current which matched best with the query. 
        // The index of descriptor current, must be matched with the index of match shortlist
        // and then the corresponding index of the current frame must be assigned to the
        // previous frame and vice versa.
        
        // Assigning to each others previous and next
        query_framepoint.next = &current_frame[good_matches[0].trainIdx];
        current_frame[good_matches[0].trainIdx].previous = &query_framepoint;
    };

    return current_frame;
};