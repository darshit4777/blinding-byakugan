#include <sharp-eye/pose_optimizer.hpp>

typedef std::vector<VisualSlamBase::KeypointWD> FeatureVector;
typedef std::vector<VisualSlamBase::Framepoint> FramepointVector;
typedef std::vector<VisualSlamBase::Frame> FrameVector;
typedef std::vector<std::pair<VisualSlamBase::KeypointWD,VisualSlamBase::KeypointWD>> MatchVector;
typedef VisualSlamBase::Camera Camera;

PoseOptimizer::PoseOptimizer(){
    // Setting all params
    parameters.minimum_depth = 0.1;
    parameters.maximum_depth = 10.0;
    parameters.maximum_reliable_depth = 100.0;
    parameters.kernel_maximum_error = 50;
    parameters.max_iterations = 1000;

    parameters.min_correspondences = 200;

    parameters.angular_delta = 0.001;
    parameters.translation_delta = 0.01;

    parameters.ignore_outliers = true;
    parameters.min_inliers = 100;

    // Setting up pose optimizer variables
    omega.setIdentity();
    b.resize(6);
    b.setZero();
    T_prev2curr.setIdentity();
    parameters.T_caml2camr.setIdentity();
    parameters.T_caml2camr.translation() << 0.0, 0.0, 0.11074;
    parameters.T_caml2camr.rotation().Identity();

    // Initializing errors
    iteration_error = 0;
    total_error = 0;
    reproj_error.setZero();

    std::cout<<"Pose Optimizer Initialized"<<std::endl;
    return;
};

void PoseOptimizer::Initialize(VisualSlamBase::Frame* current_frame_ptr,VisualSlamBase::Frame* previous_frame_ptr){
    // This needs to be called once before the OptimizeOnce can be used


    // No of measurements
    measurements = 0;
    for(int i =0; i< current_frame_ptr->points.size(); i++){
        if(current_frame_ptr->points[i].previous == NULL){
            continue;
        }
        else{
            measurements++;
        }
    };

    // Initializing the optimizer variables
    H.setZero();
    b.setZero();
    omega.setIdentity();
    T_prev2curr = previous_frame_ptr->T_cam2world;

    return;
};

void PoseOptimizer::ComputeError(VisualSlamBase::Framepoint& fp){
    /**
     * @brief Transforms the camera coordinates of points from the previous frame
     * to the current frame with an estimate of T_prev2curr. 
     * Calculates the reprojection error
     * 
    */
    
    p_caml = T_prev2curr*fp.previous->camera_coordinates;
    p_camr = parameters.T_caml2camr.inverse() * p_caml;
    iteration_error = 0;

    //if (fp.previous->landmark_set){
    //    p_caml = current_frame_ptr->T_cam2world * fp.previous->associated_landmark->world_coordinates;
    //    //increase weight for landmarks
    //    omega = 1.2 * omega;
    //    p_camr = parameters.T_caml2camr.inverse()*p_caml;
    //};

    // Checking coordinates for invalid values
    if(HasInf(p_caml) || HasInf(p_camr)){
        std::cout<<"Invalid  Camera Points - INF"<<std::endl;
        return;
        
    }
    if(p_caml.hasNaN() || p_camr.hasNaN()){
        std::cout<<"Invalid  Camera Points - NaN"<<std::endl;
        return;
    };

    // Now we project the points from the previous frame into pixel coordinates
    Eigen::Vector3d lcam_pixels,rcam_pixels;
    lcam_pixels = current_frame_ptr->camera_l.intrinsics * p_caml;
    lcam_pixels[0] = lcam_pixels[0]/lcam_pixels[2];
    lcam_pixels[1] = lcam_pixels[1]/lcam_pixels[2];
    
    rcam_pixels = current_frame_ptr->camera_r.intrinsics * p_camr;
    rcam_pixels[0] = rcam_pixels[0]/rcam_pixels[2];
    rcam_pixels[1] = rcam_pixels[1]/rcam_pixels[2];
    
    if(lcam_pixels.hasNaN() || rcam_pixels.hasNaN()){
        std::cout<<"Invalid pixels - NaN"<<std::endl;
        return;
    };
    if(rcam_pixels[0] < 0 || rcam_pixels[1] < 0){
        return;
    };
    if(lcam_pixels[0] < 0 || lcam_pixels[1] < 0){
        return;
    };

    // Calculating Reprojection Error
    reproj_error[0] = fp.keypoint_l.keypoint.pt.x - lcam_pixels[0];
    reproj_error[1] = fp.keypoint_l.keypoint.pt.y - lcam_pixels[1];

    reproj_error[2] = fp.keypoint_r.keypoint.pt.x - rcam_pixels[0];
    reproj_error[3] = fp.keypoint_r.keypoint.pt.y - rcam_pixels[1];

    const double error_squared = reproj_error.transpose()*reproj_error;
    iteration_error = error_squared;
    total_error = total_error + iteration_error;

    return;
};

bool PoseOptimizer::HasInf(Eigen::Vector3d vec){

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

void PoseOptimizer::Linearize(VisualSlamBase::Framepoint& fp){
    /**
     * @brief In this function we compute / update H, b and omega
     * 
     */

    // Setting omega for this iteration

    // Robustifying the Kernel 
    // If the reprojection error is too high, we proportionally reduce its weightage
    if(iteration_error > parameters.kernel_maximum_error){
        if(parameters.ignore_outliers){
            return;
        }
        else{
            omega = omega * parameters.kernel_maximum_error/iteration_error;
        }
    }
    else{
        if(!fp.inlier){
            fp.inlier = true;
            inliers++;
        };
        omega = omega * 1.1;
    };

    if(p_caml[2] < parameters.minimum_depth ){
        // Too close - cant use
        return;
    }

    // Now setting the weighting factor based on depth
    double translation_factor = std::min(parameters.maximum_reliable_depth/p_caml[2],1.0);

    // Calculate the jacobian
    Eigen::Matrix<double,4,6> J = FindJacobian(p_caml,p_camr,current_frame_ptr->camera_l,current_frame_ptr->camera_r,translation_factor);

    //update H and b
    H += J.transpose()*omega*J;
    b += J.transpose()*omega*reproj_error;


    return;    
};

Eigen::Matrix<double,4,6> PoseOptimizer::FindJacobian(Eigen::Vector3d& left_cam_coordinates,Eigen::Vector3d& right_cam_coordinates,Camera& camera_l,Camera& camera_r,double omega){
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

void PoseOptimizer::Solve(){
    /**
     * @brief Solve the problem H * dx = -b.
     * Also converts the output into a form that can be represented on
     * SE3
     * 
     */
    Eigen::VectorXd dx;
    dx.setZero();
    Eigen::MatrixXd identity6;
    identity6.resize(6,6);
    identity6.setIdentity();
    double damping_factor = measurements * 1;
    H = H + damping_factor * identity6;
    dx = H.ldlt().solve(-b);
    //dx = H.fullPivLu().solve(-b);

    // dx ends up being a vector with the translation variables and the rotation angles
    // The rotation angles are a normalized quaternion
        
    Eigen::Transform<double,3,2> dT;
    dT.translation().x() = dx[0];
    dT.translation().y() = dx[1];
    dT.translation().z() = dx[2];
        
    // The angles are in the form of normalized quaternion 
    Eigen::Vector3d nquaternion;
    nquaternion.x() = dx[3];
    nquaternion.y() = dx[4];
    nquaternion.z() = dx[5];
    double n = nquaternion.squaredNorm();
    Eigen::Matrix3d rot_matrix;
    if(n > 1){
        rot_matrix.setIdentity();
    }
    else{
        double w = sqrt(1 - n);
        Eigen::Quaterniond q(w,nquaternion.x(),nquaternion.y(),nquaternion.z());
        rot_matrix =  q.toRotationMatrix();
    };
    // Simplest way to assign a 3D rotation matrix
    dT.matrix().block<3,3>(0,0) = rot_matrix;

    // Update the transform
    T_prev2curr = dT * T_prev2curr;
    return;
};

void PoseOptimizer::Update(){
    /**
     * @brief Updates the T_caml2world of the current frame
     * 
     */

    // Update the pose
    current_frame_ptr->T_cam2world = previous_frame_ptr->T_cam2world * T_prev2curr.inverse();
    current_frame_ptr->T_world2cam = current_frame_ptr->T_cam2world.inverse();

    // Update the landmarks
    for(VisualSlamBase::Framepoint& fp : current_frame_ptr->points){
        fp.world_coordinates = current_frame_ptr->T_world2cam * fp.camera_coordinates;
        // Now the pose is refined - let us make them into landmarks
        if(fp.inlier && !fp.landmark_set){
            
            // Creating a new landmark
            VisualSlamBase::Landmark landmark;
            // Storing the landmark in the current local map
            lmap_ptr->associated_landmarks.push_back(landmark);
            
            // Now working with a landmark pointer once the stack pointer is assigned
            VisualSlamBase::Landmark* landmark_ptr = &lmap_ptr->associated_landmarks.back();
            landmark_ptr->world_coordinates = fp.world_coordinates;
            landmark_ptr->origin = boost::make_shared<VisualSlamBase::Framepoint>(fp);
            fp.associated_landmark = boost::make_shared<VisualSlamBase::Landmark>(*landmark_ptr);
            fp.landmark_set = true;
        };
    };
    return;
};

void PoseOptimizer::OptimizeOnce(){
    
    // Resetting the optimizer params
    H.setZero();
    b.setZero();
    omega.setIdentity();
    reproj_error.setZero();
    iteration_error = 0;

    for(VisualSlamBase::Framepoint& fp : current_frame_ptr->points){
        ComputeError(fp);
        Linearize(fp);
    };
    Solve();
    return;
}

void PoseOptimizer::Converge(){

    // We create convergence and solving criteria here
    return;
};

PoseOptimizer::~PoseOptimizer(){
    return;
};