#include<sharp-eye/visual_tracking.hpp>
#include<sophus/se3.hpp>
#include<opencv2/calib3d/calib3d.hpp>

typedef std::vector<KeypointWD> FeatureVector;
typedef std::vector<boost::shared_ptr<Framepoint>> FramepointPointerVector;
typedef std::vector<std::pair<KeypointWD,KeypointWD>> MatchVector;
typedef boost::shared_ptr<Framepoint> FramepointShared;
typedef Camera Camera;
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
    InitializeStateJacobian();

    // Optimizer
    optimizer = new PoseOptimizer;

    std::cout<<"Visual Tracking Initialized"<<std::endl;  
};

int VisualTracking::FindCorrespondences(FramepointPointerVector& previous_frame,FramepointPointerVector& current_frame){

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

    // Initialization
    for(int i =0; i < current_frame.size(); i++){
        current_frame[i]->previous = NULL;
    }
    
    // Sort Previous
    std::sort(previous_frame.begin(),previous_frame.end(),[](const FramepointShared& a,const FramepointShared& b){
        return ((a.get()->keypoint_l.keypoint.pt.y < b.get()->keypoint_l.keypoint.pt.y)||
        (a.get()->keypoint_l.keypoint.pt.y == b.get()->keypoint_l.keypoint.pt.y && a.get()->keypoint_l.keypoint.pt.x < b.get()->keypoint_l.keypoint.pt.x));
    });

    // Sort Current
    std::sort(current_frame.begin(),current_frame.end(),[](const FramepointShared& a,const FramepointShared& b){
        return ((a.get()->keypoint_l.keypoint.pt.y < b.get()->keypoint_l.keypoint.pt.y)||
        (a.get()->keypoint_l.keypoint.pt.y == b.get()->keypoint_l.keypoint.pt.y && a.get()->keypoint_l.keypoint.pt.x < b.get()->keypoint_l.keypoint.pt.x));
    });
    // We use the previous frame as the query frame

    for(int i =0; i<previous_frame.size(); i++){
        Framepoint* query_framepoint = previous_frame[i].get();

        // Break Condition
        int id_current = 0;
        std::vector<int> match_shortlist; // Framepoints found in the rectangular search region
        int ymin,ymax,xmin,xmax;
        ymin = std::max(int(query_framepoint->keypoint_l.keypoint.pt.y - 50),0);
        ymax = std::min(int(query_framepoint->keypoint_l.keypoint.pt.y + 50),img_height);

        xmin = std::max(int(query_framepoint->keypoint_l.keypoint.pt.x - 50),0);
        xmax = std::min(int(query_framepoint->keypoint_l.keypoint.pt.x + 50),img_width);
        
        // Loop to search for the top of the rectangular region
        // TODO We need to fix this line = its too shabby
        while((current_frame[id_current].get()->keypoint_l.keypoint.pt.y < ymin) && (id_current < current_frame.size())){
            id_current++;
            if(id_current >= current_frame.size()){
                break;
            };
        };
        
        // The search point is now within the rows of the rectangular region
        // We check each keypoint and see if it obeys the column constraints
        // when the lower row of the rectangular region is breached, we move 
        // to the next point

        while(current_frame[id_current].get()->keypoint_l.keypoint.pt.y < ymax){
            if(id_current >= current_frame.size()){
                break;
            }
            int x = current_frame[id_current].get()->keypoint_l.keypoint.pt.x;
            
            // Check if the keypoint is within the rectangle
            if((x < xmax) && (x>xmin)){
                // Within the rectangle
                match_shortlist.push_back(id_current);
            }
            id_current++;
            if(id_current >= current_frame.size()){
                break;
            }
        };

        if(match_shortlist.empty()){
            // This means that for the query point, there is no match in the current frame

            // We check if the queried point itself has a previous
            if(query_framepoint->previous == nullptr){
                // This point is then a truly lost point - it has no forward or backward correspondence
                lost_points.push_back(previous_frame[i]);
            }
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
            current_descriptor.push_back(current_frame[match_shortlist[i]].get()->keypoint_l.descriptor);
        };

        query_descriptor = query_framepoint->keypoint_l.descriptor;

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
        Framepoint* current_frame_ptr = current_frame[current_frame_idx].get();
        query_framepoint->next = current_frame_ptr;
        current_frame_ptr->previous = query_framepoint;
        //query_framepoint.next = boost::make_shared<Framepoint>( current_frame[current_frame_idx]);
        //current_frame[current_frame_idx].previous = boost::make_shared<Framepoint>(query_framepoint);
        correspondences++;
    };

    frame_correspondences = correspondences;
    return correspondences;
};

Eigen::Matrix<float,4,6> VisualTracking::FindJacobian(Eigen::Vector3f& left_cam_coordinates,Eigen::Vector3f& right_cam_coordinates,Camera& camera_l,Camera& camera_r,float omega){
    Eigen::Matrix<float,4,6> J;

    Eigen::Matrix<float,2,3> left_projection_derivative, right_projection_derivative;
    float fx_l,fy_l;
    float fx_r,fy_r;

    fx_l = camera_l.intrinsics(0,0);
    fy_l = camera_l.intrinsics(1,1);

    fx_r = camera_r.intrinsics(0,0);
    fy_r = camera_r.intrinsics(1,1);

    float x_l,y_l,z_l;
    float x_r,y_r,z_r;
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

    Eigen::Matrix3f hat_cam_coordinates;
    Eigen::Matrix3f identity3;
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

    Eigen::Matrix<float,3,6> J_Transform;
    J_Transform.block<3,3>(0,0) = identity3 * omega;
    J_Transform.block<3,3>(0,3) = -hat_cam_coordinates;

    J.block<2,6>(0,0) = left_projection_derivative * J_Transform;
    J.block<2,6>(2,0) = right_projection_derivative * J_Transform;

    //Eigen::Matrix<float,2,6> J_test;
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

Eigen::Transform<float,3,2> VisualTracking::EstimateIncrementalMotion(){
    optimizer->parameters.T_caml2camr = T_caml2camr;
     
    LocalMap* lmap_ptr = map.GetLastLocalMap();
    Frame* previous_frame_ptr = lmap_ptr->GetPreviousFrame();
    Frame* current_frame_ptr = lmap_ptr->GetLastFrame();
    optimizer->Initialize(current_frame_ptr,previous_frame_ptr,lmap_ptr);
    optimizer->OptimizeOnce();
    optimizer->Converge();

    return current_frame_ptr->T_world2cam;
};

void VisualTracking::CreateAndUpdateLandmarks(Frame* current_frame_ptr,LocalMap* lmap_ptr){
    //Update the landmarks

    // Loop through all the points of the current frame
    for(int i =0; i<current_frame_ptr->points.size();i++){
        Framepoint* fp = current_frame_ptr->points[i].get();

        // Now the pose is refined - let us check if we can convert it to a landmark
        if(fp->inlier){
            // Check the track length
            // TODO parameterize this 
            int threshold_track_length = 3;
            int count = 0;
            bool track_broken = false;

            // Creating a shifting pointer
            Framepoint* framepoint_ptr;
            framepoint_ptr = fp;
            
            while(!track_broken){
                
                if(framepoint_ptr->previous != nullptr){
                    // Check if the previous has a landmark associated with it
                    if(framepoint_ptr->landmark_set){
                        // Previous has a landmark associated with it - update it.
                        framepoint_ptr->associated_landmark->UpdateLandmark(current_frame_ptr->points[i]);
                        // Need to be sure here that we aren't adding the same landmark again and again
                        auto test_ptr = std::find(actively_tracked_landmarks.begin(),actively_tracked_landmarks.end(),framepoint_ptr->associated_landmark);
                        if(test_ptr == actively_tracked_landmarks.end()){
                            // Landmark hasnt been added - Add it now
                            actively_tracked_landmarks.push_back(framepoint_ptr->associated_landmark);
                        };
                        count = 0;
                        break; 
                    }
                    else{
                        count++;
                        framepoint_ptr = framepoint_ptr->previous;
                        track_broken = false;
                    };
                    
                }
                else{
                    track_broken = true;
                };
            };
            if(count > threshold_track_length){
                // Create a new landmark
                boost::shared_ptr<Landmark> landmark_ptr = boost::make_shared<Landmark>(current_frame_ptr->points[i]);
                lmap_ptr->AddLandmark(landmark_ptr);
                actively_tracked_landmarks.push_back(landmark_ptr);
                framepoint_ptr->associated_landmark = landmark_ptr;
                framepoint_ptr->landmark_set = true;
            };
        };
    };

}


void VisualTracking::RecoverLostPoints(Frame* current_frame_ptr){
    /**
     * @brief Project framepoints from the lost points vector onto the current frame
     * see if the descriptor of the projected point is close enough to that of the 
     * previous point - if it is, then we conclude that the lost point has been viewed 
     * again. We then create a new framepoint in the Frame and connect it ot the lost 
     * point
     * 
     */

    auto orb_descriptor = cv::ORB::create();

    for(int i = 0; i < lost_points.size(); i++){
        Framepoint* lost_fp_ptr = lost_points[i].get();

        // Obtain the camera coordinates
        Eigen::Vector3f camera_coordinates;
        camera_coordinates = current_frame_ptr->T_cam2world * lost_fp_ptr->world_coordinates;

        // Project the point
        Eigen::Vector3f lcam_pixels;
        lcam_pixels = camera_left.intrinsics * camera_coordinates;
        lcam_pixels[0] = lcam_pixels[0]/lcam_pixels[2];
        lcam_pixels[1] = lcam_pixels[1]/lcam_pixels[2];

        // Check if the point is valid
        if(lcam_pixels.hasNaN()){
            std::cout<<"Invalid pixels - NaN"<<std::endl;
            continue;
        };
        if(lcam_pixels[0] < 0 || lcam_pixels[1] < 0){
            continue;
        };
        if(lcam_pixels[0] > 720 || lcam_pixels[1] > 480){
            continue;
        };

        // Now we compute the descriptor at this point
         
        cv::Mat descriptor;
        cv::KeyPoint kp;
        kp.pt.x = lcam_pixels[0];
        kp.pt.y = lcam_pixels[1];
        std::vector<cv::KeyPoint> kp_vector;
        std::vector<std::vector<cv::KeyPoint>> kp_parent_vector;
        kp_vector.push_back(kp);
        orb_descriptor->detectAndCompute(current_frame_ptr->image_l,cv::noArray(),kp_vector,descriptor,true);
        
        // Now we compare the descriptor computed and the one that exists in the lost fp
        double distance = cv::norm(lost_fp_ptr->keypoint_l.descriptor,descriptor,cv::NORM_HAMMING2);
        if(distance < 6){
            // We consider this a match 
            // Create a new framepoint and connect it to this one
            boost::shared_ptr<Framepoint> fp;
            fp = boost::make_shared<Framepoint>();
            Framepoint* fp_ptr = fp.get();
            //TODO How do we add the right keypoint here
            fp_ptr->keypoint_l.descriptor = descriptor;
            fp_ptr->keypoint_l.keypoint.pt.x = lcam_pixels[0];
            fp_ptr->keypoint_l.keypoint.pt.y = lcam_pixels[0];
            fp_ptr->camera_coordinates = camera_coordinates;
            fp_ptr->inlier = false;
            fp_ptr->landmark_set = false;
            fp_ptr->world_coordinates = current_frame_ptr->T_world2cam * camera_coordinates;
            fp_ptr->previous = lost_fp_ptr;
            fp_ptr->parent_frame = current_frame_ptr;
            lost_fp_ptr->next = fp_ptr;

            // Add to the current frame
            current_frame_ptr->points.push_back(fp);

            // Remove the lost point from the list of lost points
            lost_points.erase(lost_points.begin() + i);
            // Adjusting index to account for lost point
            i--;
        }
        else{
            // The matching distance is not close enough - its not a match
            continue;
        };

    };

    delete orb_descriptor;
    return;
}

VisualTracking::ManifoldDerivative VisualTracking::CalculateMotionJacobian(Frame* current_frame_ptr,Frame* previous_frame_ptr){
    /**
     * @brief Time differentials on SE3 are calculated as deltaT = T1.inverse() * T2
     * 
     */

    Eigen::Transform<float,3,2> T2,T1;
    T1 = previous_frame_ptr->T_world2cam;
    T2 = current_frame_ptr->T_world2cam;
    state_jacobian.deltaT = T1.inverse() * T2;
    auto current_time = std::chrono::high_resolution_clock::now();
    if(state_jacobian.clock_set){
        state_jacobian.differential_interval = std::chrono::duration<float, std::milli>(state_jacobian.prediction_call - state_jacobian.previous_prediction_call).count();
    }
    else{
        std::cout<<"Warning: Debug : The differential time delta was not set, this may cause pose predictions to fail"<<std::endl;
    };
    state_jacobian.deltaT_set = true;
    return state_jacobian;

};


Eigen::Transform<float,3,2> VisualTracking::CalculatePosePrediction(Frame* frame_ptr){
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
    Eigen::Transform<float,3,2> T_predicted;
    T_predicted.setIdentity();
    if(state_jacobian.deltaT_set){
        float time_elapsed = std::chrono::duration<float, std::milli>(state_jacobian.prediction_call - state_jacobian.previous_prediction_call).count();
        float time_fraction = time_elapsed/state_jacobian.differential_interval;
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
    // Reassigning all framepoints to heap memory and adding them to the framepoint vector

    for(int i = 0; i < framepoints.size(); i++){
     boost::shared_ptr<Framepoint> fp_ptr = boost::make_shared<Framepoint>(framepoints[i]);
     framepoint_vec.push_back(fp_ptr);

    }
    return;
};

void VisualTracking::InitializeStateJacobian(){
    state_jacobian.clock_set = false;
    state_jacobian.deltaT_set = false;
    state_jacobian.deltaT.setIdentity();
    state_jacobian.prediction_interval = -1;
    state_jacobian.differential_interval = -1;
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
        map.CreateNewLocalMap();
        LocalMap* lmap_ptr = map.GetLastLocalMap();

        // Create a new frame
        Eigen::Transform<float,3,2> T;
        T.setIdentity();
        lmap_ptr->CreateNewFrame(img_l,img_r,framepoint_vec,T);
        
        // Get a raw pointer to the frame
        Frame* frame_ptr = lmap_ptr->GetLastFrame();
        
        // Framepoint initialization 
        for(int i =0; i < frame_ptr->points.size(); i++){
            Framepoint* framepoint_ptr = frame_ptr->points[i].get();
            framepoint_ptr->world_coordinates = framepoint_ptr->camera_coordinates;
            framepoint_ptr->landmark_set = false;
            framepoint_ptr->inlier = false;
            framepoint_ptr->next = NULL;
            //framepoint_ptr->associated_landmark = NULL;
            framepoint_ptr->parent_frame = frame_ptr;
        };

        // TODO : what if the framepoint vector is empty ? 
        // TODO : No need to put cameras in each frame - that's too much
        frame_ptr->camera_l = camera_left;
        frame_ptr->camera_r = camera_right;

        std::cout<<"Empty local map, new frame"<<std::endl;
        return;
    }
    else{
        // If local maps are not empty, then we are currently working on a local map.
        // - Get the current local map we are working with
        LocalMap* current_lmap = map.GetLastLocalMap();
    
        // The frames are not empty. The first frame has been set.
        // There is atleast one more frame in the local map
        // We must check if it is appropriate to continue in this local map or to create 
        // a new one. This is done by checking the correspondences between the current 
        // frame and the previous one. 

        current_lmap->CreateNewFrame(img_l,img_r,framepoint_vec);
        Frame* frame_ptr = current_lmap->GetLastFrame();
        Frame* prev_frame_ptr = current_lmap->GetPreviousFrame();
        // TODO : what if the framepoint vector is empty ? 
        frame_ptr->camera_l = camera_left;
        frame_ptr->camera_r = camera_right;

        
        // Now we check correspondences
        
        int correspondences;
        correspondences = FindCorrespondences(prev_frame_ptr->points,frame_ptr->points);
        if(correspondences == 0){
            // The track is broken. We must initialize a new local map
            std::cout<<"No correspondences found, creating new local map and new Frame"<<std::endl;
            // Deleting the last frame from the old map
            current_lmap->frames.pop_back();
            
            // Creating new local map
            map.CreateNewLocalMap();
            LocalMap* lmap_ptr = map.GetLastLocalMap();

            // Creating a new frame for the new local map
            lmap_ptr->CreateNewFrame(img_l,img_r,framepoint_vec);

            // Reassign the pointer before it gets accessed
            frame_ptr = lmap_ptr->GetLastFrame();
            prev_frame_ptr = current_lmap->GetLastFrame();
            // Setting the coordinates for the new local map
            
            // If a pose derivative is available, we can perform pose prediction to get 
            // the pose of the new local map
            if(state_jacobian.deltaT_set){

                frame_ptr->T_world2cam = CalculatePosePrediction(prev_frame_ptr);
                frame_ptr->T_cam2world = frame_ptr->T_world2cam.inverse();
                lmap_ptr->T_world2map = frame_ptr->T_world2cam;
                lmap_ptr->T_map2world = lmap_ptr->T_world2map.inverse();
            }
            else{
                // If the pose derivative is not available for some reason. 
                // We attach the pose of the latest known frame.
                std::cout<<"Warninig : Debug : Pose derivative was not available for a case where a new local map was being created"<<std::endl;
                
                frame_ptr->T_world2cam = prev_frame_ptr->T_world2cam;
                frame_ptr->T_cam2world = prev_frame_ptr->T_cam2world;
                lmap_ptr->T_world2map = prev_frame_ptr->T_world2cam;
                lmap_ptr->T_map2world = prev_frame_ptr->T_cam2world;
            };
            // Setting the frame world coordinates
            for(int i=0; i < frame_ptr->points.size(); i++){
                Framepoint* framepoint_ptr = frame_ptr->points[i].get();
                framepoint_ptr->world_coordinates = frame_ptr->T_world2cam * framepoint_ptr->camera_coordinates;
                framepoint_ptr->landmark_set = false;
                framepoint_ptr->inlier = false;
                framepoint_ptr->next = NULL;
                //framepoint_ptr->associated_landmark = NULL;
                framepoint_ptr->parent_frame = frame_ptr;
            };
            return;
        }
        else{
            // Track is not broken, we can continue operating in the same local map
            if(state_jacobian.deltaT_set){
                // The derivative is pre-calculated and available
                frame_ptr->T_world2cam = CalculatePosePrediction(prev_frame_ptr);
                frame_ptr->T_world2cam = prev_frame_ptr->T_world2cam;
                frame_ptr->T_cam2world = frame_ptr->T_world2cam.inverse();
                
                for(int i =0; i < frame_ptr->points.size(); i++){
                    Framepoint* framepoint_ptr;
                    framepoint_ptr->world_coordinates = frame_ptr->T_world2cam*framepoint_ptr->camera_coordinates;
                    framepoint_ptr->landmark_set = false;
                    framepoint_ptr->inlier = false;
                    framepoint_ptr->next = NULL;
                    //framepoint_ptr->associated_landmark = NULL;
                    framepoint_ptr->parent_frame = frame_ptr;
                };
                std::cout<<"Pose Prediction applied"<<std::endl;
            }
            else{
                // The derivative is not available

                // Get previous known frame
                frame_ptr->T_world2cam = prev_frame_ptr->T_world2cam;
                frame_ptr->T_cam2world = prev_frame_ptr->T_cam2world;
                for(int i =0; i < frame_ptr->points.size(); i++){
                    Framepoint* framepoint_ptr;
                    framepoint_ptr->world_coordinates = frame_ptr->T_world2cam * framepoint_ptr->camera_coordinates;
                    framepoint_ptr->landmark_set = false;
                    framepoint_ptr->inlier = false;
                    framepoint_ptr->next = NULL;
                    //framepoint_ptr->associated_landmark = NULL;
                    framepoint_ptr->parent_frame = frame_ptr;
                };
                std::cout<<"Pose Prediction Not applied"<<std::endl;
            };
            std::cout<<"Correspondences available, adding new frame to active local map"<<std::endl;
            return;
        };  
    };  
};

bool VisualTracking::HasInf(Eigen::Vector3f vec){
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
    
