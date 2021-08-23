#include<sharp-eye/visual_tracking.hpp>
#include<sophus/se3.hpp>
#include<opencv2/calib3d/calib3d.hpp>

typedef std::vector<VisualSlamBase::KeypointWD> FeatureVector;
typedef std::vector<VisualSlamBase::Framepoint> FramepointVector;
typedef std::vector<VisualSlamBase::Frame> FrameVector;
typedef std::vector<std::pair<VisualSlamBase::KeypointWD,VisualSlamBase::KeypointWD>> MatchVector;
typedef VisualSlamBase::Camera Camera;
typedef std::chrono::high_resolution_clock _Clock; 
typedef std::chrono::_V2::high_resolution_clock::time_point _Time;

VisualTracking::VisualTracking(Camera &cam_left,Camera &cam_right){

    // Initialize camera specifics
    camera_left = cam_left;
    camera_right = cam_right;
    img_height = 480;
    img_width = 720;
    
    matcher = cv::FlannBasedMatcher(new cv::flann::LshIndexParams(20, 10, 2));
    
    // Initialiaztions
    InitializeWorldMap();
    InitializeStateJacobian();

    // Optimizer
    optimizer = new PoseOptimizer;

    std::cout<<"Visual Tracking Initialized"<<std::endl;  
};

int VisualTracking::FindCorrespondences(FramepointVector &previous_frame,FramepointVector &current_frame){

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

    int correspondences = 0;
    
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
        ymin = std::max(int(query_framepoint.keypoint_l.keypoint.pt.y - 50),0);
        ymax = std::min(int(query_framepoint.keypoint_l.keypoint.pt.y + 50),img_height);

        xmin = std::max(int(query_framepoint.keypoint_l.keypoint.pt.x - 50),0);
        xmax = std::min(int(query_framepoint.keypoint_l.keypoint.pt.x + 50),img_width);
        
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
        cv::Mat current_descriptor;
    
        // The match shortlist is a vector of shortlisted indices from current frame

        // Assigning the current and query descriptors
        for(int i =0; i < match_shortlist.size(); i++){
            current_descriptor.push_back(current_frame[match_shortlist[i]].keypoint_l.descriptor);
        };

        query_descriptor = query_framepoint.keypoint_l.descriptor;

        matcher.knnMatch(query_descriptor,current_descriptor,knn_matches,2);
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
        int current_frame_idx = match_shortlist[good_matches[0].trainIdx];
        query_framepoint.next = &current_frame[current_frame_idx];
        current_frame[current_frame_idx].previous = &query_framepoint;
        //query_framepoint.next = boost::make_shared<VisualSlamBase::Framepoint>( current_frame[current_frame_idx]);
        //current_frame[current_frame_idx].previous = boost::make_shared<VisualSlamBase::Framepoint>(query_framepoint);
        correspondences++;
    };

    frame_correspondences = correspondences;
    return correspondences;
};

Eigen::Matrix<double,4,6> VisualTracking::FindJacobian(Eigen::Vector3d& left_cam_coordinates,Eigen::Vector3d& right_cam_coordinates,Camera& camera_l,Camera& camera_r,double omega){
    Eigen::Matrix<double,4,6> J;

    Eigen::Matrix<double,2,3> left_projection_derivative, right_projection_derivative;
    double fx_l,fy_l;
    double fx_r,fy_r;

    fx_l = camera_l.intrinsics(0,0);
    fy_l = camera_l.intrinsics(1,1);

    fx_r = camera_r.intrinsics(0,0);
    fy_r = camera_r.intrinsics(1,1);

    double x_l,y_l,z_l;
    double x_r,y_r,z_r;
    x_l = left_cam_coordinates[0];
    y_l = left_cam_coordinates[1];
    z_l = left_cam_coordinates[2];

    left_projection_derivative(0,0) = fx_l/z_l;
    left_projection_derivative(0,1) = 0.0;
    left_projection_derivative(0,2) = -fx_l*x_l/(z_l*z_l);
    left_projection_derivative(1,0) = 0.0;
    left_projection_derivative(1,1) = fy_l/z_l;
    left_projection_derivative(1,2) = -fy_l*y_l/(z_l*z_l);

    x_r = right_cam_coordinates[0];
    y_r = right_cam_coordinates[1];
    z_r = right_cam_coordinates[2];

    right_projection_derivative(0,0) = fx_r/z_r;
    right_projection_derivative(0,1) = 0.0;
    right_projection_derivative(0,2) = -fx_r*x_r/(z_r*z_r);
    right_projection_derivative(1,0) = 0.0;
    right_projection_derivative(1,1) = fy_r/z_r;
    right_projection_derivative(1,2) = -fy_r*y_r/(z_r*z_r);

    Eigen::Matrix3d hat_cam_coordinates;
    Eigen::Matrix3d identity3;
    identity3.setIdentity();
    //std::cout<<"G coordinates "<<x_l<<" "<<y_l<<" "<<" "<<z_l<<std::endl;
    hat_cam_coordinates(0,0) = 0.0;
    hat_cam_coordinates(0,1) = -2*z_l;
    hat_cam_coordinates(0,2) = 2*y_l;
    hat_cam_coordinates(1,0) = 2*z_l;
    hat_cam_coordinates(1,1) = 0.0;
    hat_cam_coordinates(1,2) = -2*x_l;
    hat_cam_coordinates(2,0) = -2*y_l;
    hat_cam_coordinates(2,1) = 2*x_l;
    hat_cam_coordinates(2,2) = 0.0;

    Eigen::Matrix<double,3,6> J_Transform;
    J_Transform.block<3,3>(0,0) = identity3 * omega;
    J_Transform.block<3,3>(0,3) = -hat_cam_coordinates;

    J.block<2,6>(0,0) = left_projection_derivative * J_Transform;
    J.block<2,6>(2,0) = right_projection_derivative * J_Transform;

    //Eigen::Matrix<double,2,6> J_test;
    //J_test(0,0) = fx_l/z_l;
    //J_test(0,1) = 0;
    //J_test(0,2) = -fx_l * x_l /(z_l*z_l);
    //J_test(0,3) = -fx_l * x_l * y_l / (z_l*z_l);
    //J_test(0,4) = fx_l * (1 + (x_l*x_l)/(z_l*z_l));
    //J_test(0,5) = -fx_l * y_l /z_l;
    //J_test(1,0) = 0;
    //J_test(1,1) = fy_l / z_l;
    //J_test(1,2) = -fy_l * y_l /(z_l*z_l);
    //J_test(1,3) = -fy_l * (1 + (y_l*y_l)/(z_l*z_l));
    //J_test(1,4) = fy_l * x_l * y_l / (z_l*z_l);
    //J_test(1,5) = fy_l * x_l /z_l;

    return J;
};

Eigen::Transform<double,3,2> VisualTracking::EstimateIncrementalMotion(VisualSlamBase::Frame &frame_ptr){
    optimizer->parameters.T_caml2camr = T_caml2camr;
    VisualSlamBase::Frame* previous_frame_ptr = GetCurrentFrame();
    VisualSlamBase::LocalMap* lmap_ptr = &map.local_maps.back();
    optimizer->Initialize(&frame_ptr,previous_frame_ptr,lmap_ptr);
    optimizer->OptimizeOnce();
    optimizer->Converge();
    
    return frame_ptr.T_world2cam;
};

VisualTracking::ManifoldDerivative VisualTracking::CalculateMotionJacobian(VisualSlamBase::Frame* current_frame_ptr,VisualSlamBase::Frame* previous_frame_ptr){
    /**
     * @brief Time differentials on SE3 are calculated as deltaT = T1.inverse() * T2
     * 
     */

    Eigen::Transform<double,3,2> T2,T1;
    T1 = previous_frame_ptr->T_world2cam;
    T2 = current_frame_ptr->T_world2cam;
    state_jacobian.deltaT = T1.inverse() * T2;
    auto current_time = std::chrono::high_resolution_clock::now();
    if(state_jacobian.clock_set){
        state_jacobian.differential_interval = std::chrono::duration<double, std::milli>(state_jacobian.prediction_call - state_jacobian.previous_prediction_call).count();
    }
    else{
        std::cout<<"Warning: Debug : The differential time delta was not set, this may cause pose predictions to fail"<<std::endl;
    };
    state_jacobian.deltaT_set = true;
    return state_jacobian;

};


Eigen::Transform<double,3,2> VisualTracking::CalculatePosePrediction(VisualSlamBase::Frame* frame_ptr){
    /**
     * @brief The method of performing pose prediction on SE3 involves a small hack
     * Tpredicted = T1 + deltaT;
     * deltaT = DT/dt * t
     * 
     * On SE3 multiplication with time translates to raising a matrix to a power 
     * In this implementation we try to find the closes whole number we can raise 
     * the matrix to and for loop it.
     */
    auto current_time = std::chrono::high_resolution_clock::now();
    Eigen::Transform<double,3,2> T_predicted;
    T_predicted.setIdentity();
    if(state_jacobian.deltaT_set){
        double time_elapsed = std::chrono::duration<double, std::milli>(state_jacobian.prediction_call - state_jacobian.previous_prediction_call).count();
        double time_fraction = time_elapsed/state_jacobian.differential_interval;
        int iterations = 0;
        if(state_jacobian.differential_interval == 0){
            iterations = 0;
        }
        else{
            iterations = std::max(int(time_elapsed/state_jacobian.differential_interval),0);
            iterations = std::min(int(time_elapsed/state_jacobian.differential_interval),0);
        };

        // Apply the transform
        for(int i = 0; i < iterations; i++){
            T_predicted = state_jacobian.deltaT * frame_ptr->T_world2cam;
        };
    };
    return T_predicted;
};

void VisualTracking::SetPredictionCallTime(){
    //TODO : Maybe these functions should be a part of the struct itself.
    auto current_time = std::chrono::_V2::high_resolution_clock::now();
    if(state_jacobian.clock_set){
        state_jacobian.previous_prediction_call = state_jacobian.prediction_call;
    }
    else{
        // For the first frame
        state_jacobian.previous_prediction_call = current_time;
        state_jacobian.clock_set = true;
    };

    state_jacobian.prediction_call = current_time;
    
    return;
};

void VisualTracking::SetFramepointVector(FramepointVector& framepoints){
    // Sets the framepoint vector
    this->framepoint_vec = framepoints;
    return;
};

VisualSlamBase::Frame* VisualTracking::GetCurrentFrame(){
    /**
     * @brief Returns a pointer to the current frame
     * 
     */
    VisualSlamBase::LocalMap* lmap_ptr;
    VisualSlamBase::Frame* frame_ptr;

    lmap_ptr = &map.local_maps.back();
    frame_ptr = &lmap_ptr->frames.back();

    return frame_ptr;
};

VisualSlamBase::Frame* VisualTracking::GetPreviousFrame(){
    /**
     * @brief Returns a pointer to the current frame
     * 
     */
    VisualSlamBase::LocalMap* lmap_ptr;
    VisualSlamBase::Frame* frame_ptr;

    lmap_ptr = &map.local_maps.back();
    int previous_frame_idx;
    previous_frame_idx = lmap_ptr->frames.size() - 2;
    frame_ptr = &lmap_ptr->frames[previous_frame_idx];

    return frame_ptr;
}

void VisualTracking::InitializeStateJacobian(){
    state_jacobian.clock_set = false;
    state_jacobian.deltaT_set = false;
    state_jacobian.deltaT.setIdentity();
    state_jacobian.prediction_interval = -1;
    state_jacobian.differential_interval = -1;
    return;
};

void VisualTracking::InitializeWorldMap(){
    map.T_world2map.setIdentity();
    map.T_world2map.setIdentity();

    std::cout<<"Initialized the World Map"<<std::endl;
    return;
};

void VisualTracking::InitializeNode(){
    /**
     * @brief Initializes a Frame or a Local map on the basis of availability of 
     * previous correspondences and pose derivatives
     * 
     */

    // Initialization for the first frame and first local map
    if(map.local_maps.empty()){
        // No local maps have been create yet - this is probably the first local map.
        VisualSlamBase::LocalMap l_map;
        // Setting the coordinates for the first local map
        l_map.T_world2map.setIdentity();
        l_map.T_map2world.setIdentity();

        // Initializing the first frame
        VisualSlamBase::Frame frame;
        // Adding points as the current framepoint vector
        frame.points = framepoint_vec;
        for(VisualSlamBase::Framepoint& framepoint : frame.points){
            framepoint.world_coordinates = framepoint.camera_coordinates;
            framepoint.landmark_set = false;
            framepoint.inlier = false;
            framepoint.next = NULL;
            framepoint.associated_landmark = NULL;

        };
        // TODO : what if the framepoint vector is empty ? 
        frame.camera_l = camera_left;
        frame.camera_r = camera_right;

        frame.image_l = img_l;
        frame.image_r = img_r;

        // Setting identity transforms for first frame
        frame.T_world2cam.setIdentity();
        frame.T_cam2world.setIdentity();

        // Add them to the object 
        l_map.frames.push_back(frame);
        map.local_maps.push_back(l_map);

        std::cout<<"Empty local map, new frame"<<std::endl;
        return;
    }
    else{
        // If local maps are not empty, then we are currently working on a local map.
        // - Get the current local map we are working with
        VisualSlamBase::LocalMap* current_lmap;
        int current_lmap_idx = map.local_maps.size() - 1;
        current_lmap = &map.local_maps[current_lmap_idx];
    
    
        // The frames are not empty. The first frame has been set.
        // There is atleast one more frame in the local map
        // We must check if it is appropriate to continue in this local map or to create 
        // a new one. This is done by checking the correspondences between the current 
        // frame and the previous one. 

        // Initialize the frame first
        VisualSlamBase::Frame frame;
        // Adding points as the current framepoint vector
        frame.points = framepoint_vec;
        // TODO : what if the framepoint vector is empty ? 
        frame.camera_l = camera_left;
        frame.camera_r = camera_right;
        frame.image_l = img_l;
        frame.image_r = img_r;
        // Now we check correspondences
        int current_frame_idx = current_lmap->frames.size() -1;
        int correspondences;
        correspondences = FindCorrespondences(current_lmap->frames[current_frame_idx].points,frame.points);
        if(correspondences == 0){
            // The track is broken. We must initialize a new local map
            std::cout<<"No correspondences found, creating new local map and new Frame"<<std::endl;
            VisualSlamBase::LocalMap l_map_new;
            
            // Setting the coordinates for the new local map
            
            // If a pose derivative is available, we can perform pose prediction to get 
            // the pose of the new local map
            if(state_jacobian.deltaT_set){

                VisualSlamBase::Frame* last_frame = GetCurrentFrame();
                frame.T_world2cam = CalculatePosePrediction(last_frame);
                
                frame.T_cam2world = frame.T_world2cam.inverse();
                l_map_new.T_world2map = frame.T_world2cam;
                l_map_new.T_map2world = l_map_new.T_world2map.inverse();
            }
            else{
                // If the pose derivative is not available for some reason. 
                // We attach the pose of the latest known frame.
                std::cout<<"Warninig : Debug : Pose derivative was not available for a case where a new local map was being created"<<std::endl;
                VisualSlamBase::Frame* last_frame = GetCurrentFrame();
                
                frame.T_world2cam = last_frame->T_world2cam;
                frame.T_cam2world = last_frame->T_cam2world;
                l_map_new.T_world2map = last_frame->T_world2cam;
                l_map_new.T_map2world = last_frame->T_cam2world;
            };
            // Setting the frame world coordinates
            for(VisualSlamBase::Framepoint& framepoint : frame.points){
                framepoint.world_coordinates = frame.T_world2cam * framepoint.camera_coordinates;
                framepoint.landmark_set = false;
                framepoint.inlier = false;
                framepoint.next = NULL;
                framepoint.associated_landmark = NULL;
            };
            l_map_new.frames.push_back(frame);
            map.local_maps.push_back(l_map_new);
            return;
        }
        else{
            // Track is not broken, we can continue operating in the same local map
            if(state_jacobian.deltaT_set){
                // The derivative is pre-calculated and available
            
                VisualSlamBase::Frame* last_frame = GetCurrentFrame();
                //frame.T_world2cam = CalculatePosePrediction(last_frame);
                frame.T_world2cam = last_frame->T_world2cam;
                frame.T_cam2world = frame.T_world2cam.inverse();
                for(VisualSlamBase::Framepoint& framepoint : frame.points){
                    framepoint.world_coordinates = frame.T_world2cam*framepoint.camera_coordinates;
                    framepoint.landmark_set = false;
                    framepoint.inlier = false;
                    framepoint.next = NULL;
                    framepoint.associated_landmark = NULL;
                };
                std::cout<<"Pose Prediction applied"<<std::endl;
            }
            else{
                // The derivative is not available

                // Get last known frame
                VisualSlamBase::Frame* last_frame = GetCurrentFrame();
                frame.T_world2cam = last_frame->T_world2cam;
                frame.T_cam2world = last_frame->T_cam2world;

                for(VisualSlamBase::Framepoint& framepoint : frame.points){
                    framepoint.world_coordinates = frame.T_world2cam * framepoint.camera_coordinates;
                    framepoint.landmark_set = false;
                    framepoint.inlier = false;
                    framepoint.next = NULL;
                    framepoint.associated_landmark = NULL;
                };
                std::cout<<"Pose Prediction Not applied"<<std::endl;
            }
            current_lmap->frames.push_back(frame);
            std::cout<<"Correspondences available, adding new frame to active local map"<<std::endl;
            return;
        };  
    };  
};

bool VisualTracking::HasInf(Eigen::Vector3d vec){
    /**
     * @brief Checks the vector to see if any of the elements have an inf
     * 
     */

    for(int i = 0; i<vec.size(); i++){
        if(std::isinf(vec[i])){
            return true;
        };
    };  
    return false;
};
    
