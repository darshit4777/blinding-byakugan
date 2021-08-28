#include <slam_datatypes/landmark.hpp>
#include <slam_datatypes/frame.hpp>
#include <slam_datatypes/framepoint.hpp>


Landmark::Landmark(boost::shared_ptr<Framepoint> fp){
    // Using a copy constructor to initialize the origin of the framepoint
    // We thus have a copy of the shared pointer

    origin = fp;
    
    world_coordinates = origin->world_coordinates;
    omega.setIdentity();
    nu.x() = 1 / world_coordinates.x();
    nu.y() = 1 / world_coordinates.y();
    nu.z() = 1 / world_coordinates.z();

    optimizer.measurement_vector.push_back(origin);
    return;
};

void Landmark::PoseOptimizer::Initialize(Eigen::Vector3f world_coordinates_){
    this->params.minimum_depth = 0.1;
    this->params.kernel_maximum_error = 1.0;
    this->params.max_iterations = 100;

    this->params.ignore_outliers = false;
    this->params.convergence_threshold = 1e-3;


    H.setZero();
    omega.setIdentity();
    b.resize(6);
    b.setZero();
    
    // Set the initial estimate of world coordinates as the original location 
    estimated_world_coordinates = world_coordinates_;

    // We might not need this
    this->params.T_caml2camr.setIdentity();
    this->params.T_caml2camr.translation() << 0.0, 0.0, 0.11074;
    this->params.T_caml2camr.rotation().Identity();

    // Initializing errors
    iteration_error = 0;
    total_error = 0;
    distance_error.setZero();
    compute_success = false;
};

void Landmark::PoseOptimizer::ComputeError(Framepoint& fp){

    /**
     * @brief We compute the error by projecting the current world coordinates
     * of the landmark to the camera coordinates of the frame where the framepoint
     * was captured.
     * 
     * The camera coordinates so generated are then projected to produce predicted 
     * pixels for frame. These pixels are considered as the "movables"
     * 
     * Reprojection error is then calculated by comparing these pixels to "fixed"
     * measurements made when the frame was captured
     * 
     */
    
    Frame* frame_ptr = fp.parent_frame.get();

    // Estimated camera coordinates in the left camera
    p_caml = frame_ptr->T_cam2world * estimated_world_coordinates;

    // Checking coordinates for invalid values
    if(HasInf(p_caml)){
        std::cout<<"Invalid  Camera Points - INF"<<std::endl;
        compute_success = false;
        return;
        
    }
    if(p_caml.hasNaN()){
        std::cout<<"Invalid  Camera Points - NaN"<<std::endl;
        compute_success = false;
        return;
    };

    // Setting omega on the basis of inverse depth
    omega = omega * 1 / fp.camera_coordinates.z();

    // Calculating the distance error - Sampled - Fixed
    distance_error = (p_caml - fp.camera_coordinates);
    error_squared = distance_error.transpose() * omega * distance_error;
    compute_success = true;
    return;

};

void Landmark::PoseOptimizer::Linearize(Framepoint& fp){
    /**
     * @brief Here we simply linearize the problem and frame the 
     * hessian and b matrices 
     * 
     */
    // Robust Kernels - Reduce importance for points which are erroneous
    if(error_squared > params.kernel_maximum_error){
        omega = omega * params.kernel_maximum_error / error_squared;
    };

    Eigen::Matrix3f J, Jt;
    J = FindJacobian(fp);
    Jt = J.transpose();

    b += Jt * omega * distance_error;
    H += Jt * omega * J;
    return;
};

void Landmark::PoseOptimizer::OptimizeOnce(){
    // Resetting the optimizer params
    H.setZero();
    b.setZero();
    omega.setIdentity();
    distance_error.setZero();
    error_squared = 0;
    iteration_error = 0;
    inliers = 0;

    for(int i =0; i < measurement_vector.size(); i++){

        Framepoint* fp = measurement_vector[i].get();
        ComputeError(*fp);
        Linearize(*fp);
    };

    Solve();
    return;
};

Eigen::Matrix3f Landmark::PoseOptimizer::FindJacobian(Framepoint& fp){
    // The jacobian for this problem is simply the rotation matrix of the frame
    Eigen::Matrix3f rot_matrix;
    rot_matrix = fp.parent_frame->T_cam2world.rotation();

    return rot_matrix;
};


void Landmark::UpdateLandmark(boost::shared_ptr<Framepoint> fp){
    /**
     * @brief Updating the position of the landmark by incorporating a new fp
     * Running an optimization, to update the position of the landmark using 
     * multiple measurements from all recorded framepoints associated with 
     * tha landmark
     */
    
    // Add the new framepoint to the list of measurements. The point will be removed
    // if the convergence from the point is not satisfactory.
    // Copying the shared pointer to get shared ownership
    boost::shared_ptr<Framepoint> fp_new = fp;
    optimizer.measurement_vector.push_back(fp_new);

    // Initialization
    optimizer.Initialize(world_coordinates);


};