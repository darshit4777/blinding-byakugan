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
    optimizer.max_inliers_recorded = 0;
    updates = 0;
    return;
};
Landmark::PoseOptimizer::PoseOptimizer(){
    return;
}

void Landmark::PoseOptimizer::Initialize(Eigen::Vector3f world_coordinates_){
    this->params.minimum_depth = 0.1;
    this->params.kernel_maximum_error = 1.0;
    this->params.max_iterations = 100;

    this->params.ignore_outliers = false;
    this->params.convergence_threshold = 1e-3;


    H.setZero();
    omega.setIdentity();
    b.setZero();
    
    // Set the initial estimate of world coordinates as the original location 
    estimated_world_coordinates = world_coordinates_;

    // Initializing errors
    iteration_error = 0;
    total_error = 0;
    distance_error.setZero();
    compute_success = false;
};

void Landmark::PoseOptimizer::ComputeError(Framepoint& fp){

    /**
     * @brief We compute the error by transforming the world coordinates of the 
     * landmark to camera coordinates of the parent frame of the framepoint.
     * These estimated coordinates are then compared to the measured coordinates
     * of the fp, ie the camera coordinates of the fp that corresponds to the 
     * landmark.
     */
    
    Frame* frame_ptr = fp.parent_frame;

    // Estimated camera coordinates in the left camera
    p_caml = frame_ptr->T_cam2world * estimated_world_coordinates;

    // Checking if the point is an inlier or outlier
    if(p_caml.z() < float(0.0)){
        // For negative depth we consider the point as an outlier
        outliers++;
    };

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
    
    // Initializing the inliers and outliers
    inliers = 0;
    outliers = 0;

    for(int i =0; i < measurement_vector.size(); i++){

        Framepoint* fp = measurement_vector[i].get();
        ComputeError(*fp);
        Linearize(*fp);

        iteration_error = iteration_error + error_squared;
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

void Landmark::PoseOptimizer::Solve(){
    /**
     * @brief Solve the equation Dx = - b / H;
     * 
     */

    Eigen::Vector3f dx;
    dx.setZero();

    Eigen::Matrix3f identity3;
    identity3.setIdentity();

    // Unity damping factor
    H = H + identity3;
    dx = H.ldlt().solve(-b);

    // dx ends up being a vector of (dx ,dy and dx)

    // Updating the world coordinates
    estimated_world_coordinates = estimated_world_coordinates + dx;

    return;
};

void Landmark::PoseOptimizer::Converge(){
    /**
     * @brief Combine all the methods together and run it in a loop till convergence
     * 
     */

    total_error = 0;
    float previous_error = 0;
    float error_delta = 0;
    int iteration_count = 0;
    bool converged = false;

    while(!converged){
        OptimizeOnce();
        total_error = total_error + iteration_error;

        // Convergence criterion
        error_delta = iteration_error - previous_error;
        previous_error = iteration_error;
        if(error_delta < 0.1 || (iteration_count > this->params.max_iterations)){
            converged = true;
        }
        iteration_count++;
    };
    return;
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
    optimizer.Converge();

    // Checking if the estimate if of a good quality
    optimizer.inliers = optimizer.measurement_vector.size() - optimizer.outliers;
    assert(optimizer.inliers > 0);

    if(optimizer.inliers > optimizer.max_inliers_recorded){
        // Updating the world coordinates
        world_coordinates = optimizer.estimated_world_coordinates;
        optimizer.max_inliers_recorded = optimizer.inliers;
        updates++;
    }
    else if (optimizer.inliers < optimizer.outliers){
        // The optimization has failed we can choose to retain the previous estimate
        // This is very poor condition where the number of outliers is more than the
        // number of inliers. We choose to take a simple average here
        std::cout<<"Not enough inliers - Averaging"<<std::endl;
        Eigen::Vector3f world_coordinates_accumulated;
        world_coordinates_accumulated.setZero();
        for(int i =0; i < optimizer.measurement_vector.size(); i++){
            Framepoint* fp = optimizer.measurement_vector[i].get();
            world_coordinates_accumulated = world_coordinates_accumulated + fp->world_coordinates;
        }
        world_coordinates_accumulated = world_coordinates_accumulated / optimizer.measurement_vector.size();
        world_coordinates = world_coordinates_accumulated;
    }
    else{
        // This is the third case, wherein the number of inliers is better than outliers,
        // but not better than previous updates - in this case we simply dont update.
    };

    return;

};
Landmark::~Landmark(){
    return;
};

Landmark::PoseOptimizer::~PoseOptimizer(){
    return;
};
bool Landmark::PoseOptimizer::HasInf(Eigen::Vector3f &vec){

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