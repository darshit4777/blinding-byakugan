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

void Landmark::UpdateLandmark(boost::shared_ptr<Framepoint> fp){
    /**
     * @brief Updating the position of the landmark by incorporating a new fp
     * Running an optimization, to update the position of the landmark using 
     * multiple measurements from all recorded framepoints associated with 
     * tha landmark
     */

    // Initialization

}