#pragma once
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include <g2o/core/optimizable_graph.h>
#include <boost/smart_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <slam_datatypes/framepoint.hpp>
#include <slam_datatypes/camera.hpp>
// Forward Declaration 
struct Framepoint;
class Landmark{
    // Landmark hold the information of multiple framepoints and their world location
    public:
    Eigen::Vector3f world_coordinates;
    boost::shared_ptr<Framepoint> origin;
    Eigen::Matrix3f omega;
    Eigen::Vector3f nu;
    std::vector<boost::shared_ptr<Framepoint>> measurement_vector;

    class PoseOptimizer{
        public:    
        struct optimization_params{
            bool ignore_outliers;
            float kernel_maximum_error;
            float minimum_depth;
            float maximum_depth;
            float maximum_reliable_depth;
            int max_iterations;

            // Movement Thresholds
            float angular_delta;
            float translation_delta;

            // Correspondences
            int min_correspondences;

            // Inliers
            int min_inliers;

            // Inter camera transform
            Eigen::Transform<float,3,2> T_caml2camr;

        };
        static optimization_params params;
        
        // Error
        float iteration_error;
        float total_error;
    
        // Optimization variables
        Eigen::Transform<float,3,2> T_prev2curr;
        Eigen::Matrix<float,6,6> H;
        Eigen::VectorXf b;
        Eigen::Matrix4f omega;
        Eigen::Vector4f reproj_error;
        float translation_factor;

        // Inliers
        int measurements;
        int inliers;

        // Camera Coordinates
        Eigen::Vector3f p_caml,p_camr;
        
        private:
        bool compute_success;

        public:
        /**
         * @brief Creates a pose optimizer object and initializes all parameters
         * 
         */
        PoseOptimizer();

        ~PoseOptimizer();

        /**
         * @brief Initializes all the pose optimizer variables
         * 
         */
        void Initialize();

        /**
         * @brief Runs the optimization loop once
         * 
         */
        void OptimizeOnce();

        /**
         * @brief Uses the optimize once along with convergence conditions
         * to generate a solution
         * 
         */
        void Converge();

        private:
        /**
         * @brief Extracts the camera coordinates from previous frame
         * checks for invalid points and then computes the reprojection error
         * 
         */
        void ComputeError(Framepoint fp);
    
        /**
         * @brief Assembles the H, b and omega matrices
         * 
         */
        void Linearize(Framepoint fp);

        /**
         * @brief Solves for Dx and transforms it into the SE3 form
         * 
         */
        void Solve();

        /**
         * @brief Updates the T_prev2current matrix and ensures the rotations 
         * are correct
         * 
         */
        void Update();

        bool HasInf(Eigen::Vector3f vec);

        Eigen::Matrix<float,4,6> FindJacobian(Eigen::Vector3f& left_cam_coordinates,Eigen::Vector3f& right_cam_coordinates,Camera& camera_l,Camera& camera_r,float omega);

    };

    struct parameters{
        int min_track_length;
    };
    static parameters params;
    /**
     * @brief Construct a new Landmark object
     * 
     */
    Landmark(boost::shared_ptr<Framepoint> fp);

    /**
     * @brief Update the landmark pose estimate with a new measurement
     * 
     * @param fp 
     */
    void UpdateLandmark(boost::shared_ptr<Framepoint> fp);

    ~Landmark();
};