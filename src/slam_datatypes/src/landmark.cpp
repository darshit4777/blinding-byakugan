#include <slam_datatypes/landmark.hpp>


Landmark::Landmark(boost::shared_ptr<Framepoint> fp){
    // Using a copy constructor to initialize the origin of the framepoint
    // We thus have a copy of the shared pointer

    origin = fp;
    
    world_coordinates = origin->world_coordinates;
    omega.setIdentity();
    nu.x() = 1 / world_coordinates.x();
    nu.y() = 1 / world_coordinates.y();
    nu.z() = 1 / world_coordinates.z();

    measurement_vector.push_back(origin);
    return;
};

void Landmark::PoseOptimizer::Initialize(Eigen::Vector3f world_coordinates_){
    this->params.minimum_depth = 0.1;
    this->params.kernel_maximum_error = 200;
    this->params.max_iterations = 100;

    this->params.ignore_outliers = false;
    this->params.convergence_threshold = 1e-3;


    omega.setIdentity();
    b.resize(6);
    b.setZero();
    
    // Set the initial estimate of world coordinates as the original location 
    world_coordinates = world_coordinates_;

    // We might not need this
    this->params.T_caml2camr.setIdentity();
    this->params.T_caml2camr.translation() << 0.0, 0.0, 0.11074;
    this->params.T_caml2camr.rotation().Identity();

    // Initializing errors
    iteration_error = 0;
    total_error = 0;
    reproj_error.setZero();

    compute_success = false;
};




void Landmark::UpdateLandmark(boost::shared_ptr<Framepoint> fp){
    /**
     * @brief Updating the position of the landmark by incorporating a new fp
     * Running an optimization, to update the position of the landmark using 
     * multiple measurements from all recorded framepoints associated with 
     * tha landmark
     */

    // Initialization

};