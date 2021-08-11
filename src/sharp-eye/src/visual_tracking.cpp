#include<sharp-eye/visual_tracking.hpp>
#include<sophus/se3.hpp>
#include<opencv2/calib3d/calib3d.hpp>
#include<g2o/solvers/eigen/linear_solver_eigen.h>

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
        query_framepoint.next = boost::make_shared<VisualSlamBase::Framepoint>( current_frame[good_matches[0].trainIdx]);
        current_frame[good_matches[0].trainIdx].previous = boost::make_shared<VisualSlamBase::Framepoint>(query_framepoint);
    };

    return current_frame;
};

Eigen::Matrix<double,4,6> VisualTracking::FindJacobian(Eigen::Vector3d& left_cam_coordinates,Eigen::Vector3d& right_cam_coordinates,Camera& camera_l,Camera& camera_r){
    Eigen::Matrix<double,4,6> J;

    Eigen::Matrix<double,2,3> left_projection_derivative, right_projection_derivative;
    double fx_l,fy_l;
    double fx_r,fy_r;

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

    hat_cam_coordinates(0,0) = 0.0;
    hat_cam_coordinates(0,1) = -2*z_l;
    hat_cam_coordinates(0,2) = -2*y_l;
    hat_cam_coordinates(1,0) = 2*z_l;
    hat_cam_coordinates(1,1) = 0.0;
    hat_cam_coordinates(1,2) = -2*x_l;
    hat_cam_coordinates(2,0) = -2*y_l;
    hat_cam_coordinates(2,1) = 2*x_l;
    hat_cam_coordinates(2,2) = 0.0;
    Eigen::Matrix<double,3,6> J_Transform;
    J_Transform.block<3,3>(0,0) = identity3;
    J_Transform.block<3,3>(0,3) = hat_cam_coordinates;

    J.block<2,6>(0,0) = left_projection_derivative * J_Transform;
    J.block<2,6>(2,0) = right_projection_derivative * J_Transform;

    return J;
};

Eigen::Transform<double,3,2> VisualTracking::EstimateIncrementalMotion(VisualSlamBase::Frame* frame_ptr){
    // Configuring the Optimization Problem
    const bool ignore_outliers = true;
    const double kernel_maximum_error = 20;
    const double close_depth = 3.5;
    const double maximum_depth = 5.0;
    const int number_of_iterations = 20;

    for (int i = 0; i < number_of_iterations; i++) {
        //Initialize least squares components
        
        Eigen::Matrix<double,6,6> H; //< Hessian
        
        Eigen::VectorXd b;
        b.resize(6);

        Eigen::Matrix4d omega; //< Information matrix 
        omega.setIdentity();

        //loop over all framepoints
        for (VisualSlamBase::Framepoint& fp : frame_ptr->points){

            if (fp.previous == NULL){
                continue;
            };

            // Transforming the corresponding points from the previous frame to current
            // frame camera coordinates
            Eigen::Vector3d p_caml = frame_ptr->T_world2cam*fp.previous->world_coordinates;
            Eigen::Vector3d p_camr = frame_ptr->T_caml2camr.inverse()*p_caml;
            
            // TODO : Use landmark position estimates
            //preferably use landmark position estimate
            //if (fp->landmark()) {
            //p_c = T_w2c*fp->landmark()->p_w;
            ////increase weight for landmarks
            //omega = ..;
            //}

            // Now we project the points from the previous frame into pixel coordinates
            Eigen::Vector3d lcam_pixels,rcam_pixels;
            lcam_pixels = frame_ptr->camera_l.intrinsics * p_caml;
            lcam_pixels[0] = lcam_pixels[0]/lcam_pixels[2];
            lcam_pixels[1] = lcam_pixels[1]/lcam_pixels[2];

            rcam_pixels = frame_ptr->camera_r.intrinsics * frame_ptr->T_caml2camr.inverse() * p_camr;
            rcam_pixels[0] = lcam_pixels[0]/lcam_pixels[2];
            rcam_pixels[1] = lcam_pixels[1]/lcam_pixels[2];

            // Calculating Reprojection Error
            Eigen::Vector4d reproj_error;
            reproj_error[0] = fp.keypoint_l.keypoint.pt.x - lcam_pixels[0];
            reproj_error[1] = fp.keypoint_l.keypoint.pt.y - lcam_pixels[1];

            reproj_error[2] = fp.keypoint_r.keypoint.pt.x - rcam_pixels[0];
            reproj_error[3] = fp.keypoint_r.keypoint.pt.y - rcam_pixels[1];

            const double error_squared = reproj_error.transpose()*reproj_error;

            // Robustify the Kernel
            if(error_squared > kernel_maximum_error){
                if(ignore_outliers){
                    continue;
                }
                else{
                    omega = omega * kernel_maximum_error/error_squared;
                }
            }
            else{
                fp.inlier = true;
            };
            
            // Calculate the jacobian
            Eigen::Matrix<double,4,6> J = FindJacobian(p_caml,p_camr,frame_ptr->camera_l,frame_ptr->camera_r);

            // Adjusting for points that are too close or too far
            if(p_caml[2] < close_depth ){
                // Too close
                omega = omega * (close_depth - p_caml[2])/close_depth;
            }
            else{
                omega = omega * (maximum_depth - p_caml[2])/maximum_depth;
                //disable contribution to translational error
                //WHY!!?
                Eigen::Matrix3d zeros;
                zeros.setZero();
                J.block<3,3>(0,0) = zeros;
            };

            //update H and b
            H += J.transpose()*omega*J;
            b += J.transpose()*omega*reproj_error;

        }
        //compute (Identity-damped) solution
        Eigen::VectorXd dx;
        g2o::LinearSolverEigen<Eigen::Matrix<double,6,6>> solver;
        
        Eigen::MatrixXd identity6;
        identity6.resize(6,6);
        identity6.setIdentity();

        H = H + identity6;
        dx = H.ldlt().solve(-b);

        // dx ends up being a vector with the translation variables and the rotation angles
        // The rotation angles are most probably rodrigues angles
        
        Eigen::Transform<double,3,2> dT;
        dT.translation().x() = dx[0];
        dT.translation().y() = dx[1];
        dT.translation().z() = dx[2];
        
        // The angles are in the form of rodrigues angels 
        std::vector<double> rvec;
        rvec.push_back(dx[3]);
        rvec.push_back(dx[4]);
        rvec.push_back(dx[5]);
        cv::Mat rotation_matrix;
        Eigen::Matrix3d eigen_rot_matrix;
        cv::Rodrigues(rvec,rotation_matrix);

        // Now converting the rotation matrix to Eigen Matrix
        eigen_rot_matrix(0,0) = rotation_matrix.at<double>(0,0);
        eigen_rot_matrix(0,1) = rotation_matrix.at<double>(0,1);
        eigen_rot_matrix(0,2) = rotation_matrix.at<double>(0,2);
        eigen_rot_matrix(1,0) = rotation_matrix.at<double>(1,0);
        eigen_rot_matrix(1,1) = rotation_matrix.at<double>(1,1);
        eigen_rot_matrix(1,2) = rotation_matrix.at<double>(1,2);
        eigen_rot_matrix(2,0) = rotation_matrix.at<double>(2,0);
        eigen_rot_matrix(2,1) = rotation_matrix.at<double>(2,1);
        eigen_rot_matrix(2,2) = rotation_matrix.at<double>(2,2);

        // Simplest way to assign a 3D rotation matrix
        dT.matrix().block<3,3>(0,0) = eigen_rot_matrix;

        frame_ptr->T_world2cam = dT*frame_ptr->T_world2cam;
        frame_ptr->T_cam2world = frame_ptr->T_world2cam.inverse();
    }

    return frame_ptr->T_cam2world;
};

