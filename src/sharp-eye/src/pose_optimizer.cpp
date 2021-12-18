#include <sharp-eye/pose_optimizer.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

typedef std::vector<KeypointWD> FeatureVector;
typedef std::vector<Framepoint> FramepointVector;
typedef std::vector<Frame> FrameVector;
typedef std::vector<std::pair<KeypointWD, KeypointWD>> MatchVector;
typedef Camera Camera;

PoseOptimizer::PoseOptimizer()
{
    // Setting all params
    parameters.minimum_depth = 0.1;
    parameters.maximum_depth = 10.0;
    parameters.maximum_reliable_depth = 10.0;
    parameters.kernel_maximum_error = 200;
    parameters.solver_maximum_error = 10;
    parameters.max_iterations = 100;

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

    // Initializing CV Window Strings
    left_cam = "Left Cam Window";
    right_cam = "Right Cam Window";

    // Creating named windows
    //cv::namedWindow(left_cam,cv::WindowFlags::WINDOW_AUTOSIZE);
    //cv::namedWindow(right_cam,cv::WindowFlags::WINDOW_AUTOSIZE);

    compute_success = false;

    std::cout << "Pose Optimizer Initialized" << std::endl;
    return;
};

void PoseOptimizer::Initialize(Frame *curr_frame_ptr, Frame *prev_frame_ptr, LocalMap *local_map_ptr)
{
    // This needs to be called once before the OptimizeOnce can be used

    current_frame_ptr = curr_frame_ptr;
    previous_frame_ptr = prev_frame_ptr;
    lmap_ptr = local_map_ptr;
    std::cout << "No of points" << std::endl;
    std::cout << current_frame_ptr->points.size() << std::endl;

    // No of measurements
    measurements = 0;
    for (int i = 0; i < current_frame_ptr->points.size(); i++)
    {
        // TODO : Fix this statement - too shabby
        if (current_frame_ptr->points[i].get()->previous == NULL)
        {
            continue;
        }
        else
        {
            measurements++;
        }
    };

    // Initializing the optimizer variables
    H.setZero();
    b.setZero();
    omega.setIdentity();
    //T_prev2curr = previous_frame_ptr->T_cam2world;
    T_prev2curr.setIdentity();
    inliers = 0;
    return;
};

float PoseOptimizer::ComputeError(Framepoint *fp, bool check_landmark)
{
    /**
     * @brief Transforms the camera coordinates of points from the previous frame
     * to the current frame with an estimate of T_prev2curr. 
     * Calculates the reprojection error
     * 
    */

    if (fp->previous == NULL)
    {
        compute_success = false;
        return -1.0;
    }

    //std::vector<Framepoint> fp_1,fp_2;
    //fp_1.push_back(*fp);
    //fp_2.push_back(*fp->previous);
    //VisualizeFramepointComparision(fp_1,current_frame_ptr->image_l,fp_2,current_frame_ptr->image_l);

    p_caml = T_prev2curr * fp->previous->camera_coordinates;
    p_camr = T_prev2curr * parameters.T_caml2camr.inverse() * fp->previous->camera_coordinates;

    
    if (fp->previous->associated_landmark && check_landmark)
    {
        // We first convert the landmark world coordinates to previous camera coordinates
        Eigen::Vector3f p_caml_prev;
        p_caml_prev = fp->previous->parent_frame->T_cam2world * fp->previous->associated_landmark->world_coordinates;
        p_caml = T_prev2curr * p_caml_prev;
        p_camr = T_prev2curr * parameters.T_caml2camr.inverse() * p_caml_prev;

        //increase weight for landmarks
        omega = 1.2 * omega;
    };

    // Checking coordinates for invalid values
    if (HasInf(p_caml) || HasInf(p_camr))
    {
        std::cout << "Invalid  Camera Points - INF" << std::endl;
        compute_success = false;
        return -1.0;
    }
    if (p_caml.hasNaN() || p_camr.hasNaN())
    {
        std::cout << "Invalid  Camera Points - NaN" << std::endl;
        compute_success = false;
        return -1.0;
    };

    // Now we project the points from the previous frame into pixel coordinates
    Eigen::Vector3f lcam_pixels, rcam_pixels;
    lcam_pixels = current_frame_ptr->camera_l.intrinsics * p_caml;
    lcam_pixels[0] = lcam_pixels[0] / lcam_pixels[2];
    lcam_pixels[1] = lcam_pixels[1] / lcam_pixels[2];

    rcam_pixels = current_frame_ptr->camera_r.intrinsics * p_camr;
    rcam_pixels[0] = rcam_pixels[0] / rcam_pixels[2];
    rcam_pixels[1] = rcam_pixels[1] / rcam_pixels[2];

    if (lcam_pixels.hasNaN() || rcam_pixels.hasNaN())
    {
        std::cout << "Invalid pixels - NaN" << std::endl;
        compute_success = false;
        return -1.0;
    };
    if (rcam_pixels[0] < 0 || rcam_pixels[1] < 0)
    {
        compute_success = false;
        return -1.0;
    };
    if (lcam_pixels[0] < 0 || lcam_pixels[1] < 0)
    {
        compute_success = false;
        return -1.0;
    };
    if (lcam_pixels[0] > 720 || lcam_pixels[1] > 480)
    {
        compute_success = false;
        return -1.0;
    };
    if (rcam_pixels[0] > 720 || rcam_pixels[1] > 480)
    {
        compute_success = false;
        return -1.0;
    };

    // Calculating Reprojection Error - Order is important - (Sampled-Fixed)
    reproj_error[0] = lcam_pixels[0] - fp->keypoint_l.keypoint.pt.x;
    reproj_error[1] = lcam_pixels[1] - fp->keypoint_l.keypoint.pt.y;

    reproj_error[2] = rcam_pixels[0] - fp->keypoint_r.keypoint.pt.x;
    reproj_error[3] = rcam_pixels[1] - fp->keypoint_r.keypoint.pt.y;
    //reproj_error[2] = 0.0;
    //reproj_error[3] = 0.0;

    float error_squared = reproj_error.transpose() * reproj_error;
    if (error_squared < parameters.kernel_maximum_error)
    {
        // Drawing it on the images for visualization
        /// Drawing the fixed points
        //cv::circle(_img_left,fp->keypoint_l.keypoint.pt,3,cv::Scalar(255,0,0),CV_FILLED);
        //cv::circle(_img_right,fp->keypoint_r.keypoint.pt,3,cv::Scalar(255,0,0),CV_FILLED);

        /// Drawing the moving points
        //cv::Point2d l_point, r_point;
        //l_point.x = lcam_pixels[0];
        //l_point.y = lcam_pixels[1];
        //cv::circle(_img_left,l_point,3,cv::Scalar(0,0,255),CV_FILLED);

        //r_point.x = rcam_pixels[0];
        //r_point.y = rcam_pixels[1];
        //cv::circle(_img_right,r_point,3,cv::Scalar(0,0,255),CV_FILLED);
    };

    iteration_error = iteration_error + error_squared;

    compute_success = true;
    return error_squared;
};

bool PoseOptimizer::HasInf(Eigen::Vector3f vec)
{

    /**
     * @brief Checks the vector to see if any of the elements have an inf
     * 
     */

    for (int i = 0; i < vec.size(); i++)
    {
        if (std::isinf(vec[i]))
        {
            return true;
        };
    };
    return false;
};

void PoseOptimizer::Linearize(Framepoint *fp)
{
    /**
     * @brief In this function we compute / update H, b and omega
     * 
     */

    // Setting omega for this iteration

    // Robustifying the Kernel
    // If the reprojection error is too high, we proportionally reduce its weightage

    if (!compute_success)
    {
        return;
    };
    float error_squared = reproj_error.transpose() * reproj_error;
    if (error_squared > parameters.kernel_maximum_error)
    {
        if (parameters.ignore_outliers)
        {
            return;
        }
        else
        {
            omega = omega * parameters.kernel_maximum_error / error_squared;
        }
    }
    else
    {
        inliers++;

        if (!fp->inlier)
        {
            fp->inlier = true;
        };
    };

    if (p_caml[2] < parameters.minimum_depth)
    {
        // Too close - cant use
        return;
    }

    // Now setting the weighting factor based on depth
    float translation_factor = std::min(parameters.maximum_reliable_depth / p_caml[2], float(1.0));

    // Calculate the jacobian
    Eigen::Matrix<float, 4, 6> J = FindJacobian(p_caml, p_camr, current_frame_ptr->camera_l, current_frame_ptr->camera_r, translation_factor);

    //update H and b
    H += J.transpose() * omega * J;
    b += J.transpose() * omega * reproj_error;

    return;
};

Eigen::Matrix<float, 4, 6> PoseOptimizer::FindJacobian(Eigen::Vector3f &left_cam_coordinates, Eigen::Vector3f &right_cam_coordinates, Camera &camera_l, Camera &camera_r, float omega)
{
    Eigen::Matrix<float, 4, 6> J;

    Eigen::Matrix<float, 2, 3> left_projection_derivative, right_projection_derivative;
    float fx_l, fy_l;
    float fx_r, fy_r;

    fx_l = camera_l.intrinsics(0, 0);
    fy_l = camera_l.intrinsics(1, 1);

    fx_r = camera_r.intrinsics(0, 0);
    fy_r = camera_r.intrinsics(1, 1);

    float x_l, y_l, z_l;
    float x_r, y_r, z_r;
    x_l = left_cam_coordinates[0];
    y_l = left_cam_coordinates[1];
    z_l = left_cam_coordinates[2];

    left_projection_derivative(0, 0) = fx_l / z_l;
    left_projection_derivative(0, 1) = 0.0;
    left_projection_derivative(0, 2) = -fx_l * x_l / (z_l * z_l);
    left_projection_derivative(1, 0) = 0.0;
    left_projection_derivative(1, 1) = fy_l / z_l;
    left_projection_derivative(1, 2) = -fy_l * y_l / (z_l * z_l);

    x_r = right_cam_coordinates[0];
    y_r = right_cam_coordinates[1];
    z_r = right_cam_coordinates[2];

    right_projection_derivative(0, 0) = fx_r / z_r;
    right_projection_derivative(0, 1) = 0.0;
    right_projection_derivative(0, 2) = -fx_r * x_r / (z_r * z_r);
    right_projection_derivative(1, 0) = 0.0;
    right_projection_derivative(1, 1) = fy_r / z_r;
    right_projection_derivative(1, 2) = -fy_r * y_r / (z_r * z_r);

    Eigen::Matrix3f hat_cam_coordinates;
    Eigen::Matrix3f identity3;
    identity3.setIdentity();
    //std::cout<<"G coordinates "<<x_l<<" "<<y_l<<" "<<" "<<z_l<<std::endl;
    hat_cam_coordinates(0, 0) = 0.0;
    hat_cam_coordinates(0, 1) = -2 * z_l;
    hat_cam_coordinates(0, 2) = 2 * y_l;
    hat_cam_coordinates(1, 0) = 2 * z_l;
    hat_cam_coordinates(1, 1) = 0.0;
    hat_cam_coordinates(1, 2) = -2 * x_l;
    hat_cam_coordinates(2, 0) = -2 * y_l;
    hat_cam_coordinates(2, 1) = 2 * x_l;
    hat_cam_coordinates(2, 2) = 0.0;

    Eigen::Matrix<float, 3, 6> J_Transform;
    J_Transform.block<3, 3>(0, 0) = identity3 * omega;
    J_Transform.block<3, 3>(0, 3) = -hat_cam_coordinates;

    J.block<2, 6>(0, 0) = left_projection_derivative * J_Transform;
    J.block<2, 6>(2, 0) = right_projection_derivative * J_Transform;

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

void PoseOptimizer::Solve()
{
    /**
     * @brief Solve the problem H * dx = -b.
     * Also converts the output into a form that can be represented on
     * SE3
     * 
     */
    Eigen::VectorXf dx;
    dx.setZero();
    Eigen::MatrixXf identity6;
    identity6.resize(6, 6);
    identity6.setIdentity();
    float damping_factor = measurements * 5;
    H = H + damping_factor * identity6;
    dx = H.ldlt().solve(-b);
    //dx = H.fullPivLu().solve(-b);

    // dx ends up being a vector with the translation variables and the rotation angles
    // The rotation angles are a normalized quaternion

    Eigen::Transform<float, 3, 2> dT;
    dT.setIdentity();
    dT.translation().x() = dx[0];
    dT.translation().y() = dx[1];
    dT.translation().z() = dx[2];

    // The angles are in the form of normalized quaternion
    Eigen::Vector3f nquaternion;
    nquaternion.x() = dx[3];
    nquaternion.y() = dx[4];
    nquaternion.z() = dx[5];
    float n = nquaternion.squaredNorm();
    Eigen::Matrix3f rot_matrix;
    if (n > 1)
    {
        rot_matrix.setIdentity();
    }
    else
    {
        float w = sqrt(1 - n);
        Eigen::Quaternionf q(w, nquaternion.x(), nquaternion.y(), nquaternion.z());
        rot_matrix = q.toRotationMatrix();
    };
    // Simplest way to assign a 3D rotation matrix
    dT.matrix().block<3, 3>(0, 0) = rot_matrix;

    // Update the transform
    T_prev2curr = dT * T_prev2curr;
    return;
};

void PoseOptimizer::Update()
{
    /**
     * @brief Updates the T_caml2world of the current frame
     * 
     */

    // T_prev2curr is actually T_curr2prev
    // Update the pose
    current_frame_ptr->T_cam2world = T_prev2curr * previous_frame_ptr->T_cam2world;
    current_frame_ptr->T_world2cam = current_frame_ptr->T_cam2world.inverse();

    // Update the world coordinates of all the points
    for (int i = 0; i < current_frame_ptr->points.size(); i++)
    {
        Framepoint *fp = current_frame_ptr->points[i].get();
        fp->world_coordinates = current_frame_ptr->T_world2cam * fp->camera_coordinates;
    };
    //// The pose optimization is complete here - we relase the images from the previous frame
    previous_frame_ptr->image_l.release();
    previous_frame_ptr->image_r.release();
    return;
};

void PoseOptimizer::OptimizeOnce(Frame *frame_ptr)
{

    // Resetting the optimizer params
    H.setZero();
    b.setZero();
    omega.setIdentity();
    reproj_error.setZero();
    iteration_error = 0;
    inliers = 0;

    for (int i = 0; i < frame_ptr->points.size(); i++)
    {
        Framepoint *fp_ptr = frame_ptr->points[i].get();
        ComputeError(fp_ptr);
        Linearize(fp_ptr);
    };
    Solve();
    return;
}

void PoseOptimizer::Converge()
{

    // We create convergence and solving criteria here
    float previous_error = 0;
    float error_delta = 0;
    for (int i = 0; i < parameters.max_iterations; i++)
    {
        //TODO : This needs work
        OptimizeOnce(current_frame_ptr);
        std::cout << "No of inliers are " << inliers << std::endl;
        if (inliers < parameters.min_inliers)
        {
            parameters.ignore_outliers = false;
        }
        else
        {
            parameters.ignore_outliers = true;
        }

        //std::cout<<"Iteration Error "<<iteration_error<<std::endl;
        error_delta = fabs(iteration_error - previous_error);
        previous_error = iteration_error;

        //std::cout<<"Error Delta "<<error_delta<<std::endl;

        if ((error_delta < 1e-5) && (iteration_error < parameters.solver_maximum_error))
        {
            std::cout << "Converged after " << i << " iterations" << std::endl;
            Update();
            return;
        }
    };
    Update();
    return;
};

PoseOptimizer::~PoseOptimizer()
{
    cv::destroyAllWindows();
    return;
};
void PoseOptimizer::VisualizeFramepoints(FramepointVector fp_vec, cv::Mat &image, int cam, cv::Scalar color = cv::Scalar(0, 0, 255))
{
    /**
     * @brief Draw framepoints on an image
     * 
     */
    // Convert the image into a RGB image
    // create 8bit color image. IMPORTANT: initialize image otherwise it will result in 32F
    cv::Mat img_rgb(image.size(), CV_8UC3);

    // convert grayscale to color image
    cv::cvtColor(image, img_rgb, CV_GRAY2RGB);

    //keypoint = fp.keypoint_l.keypoint;

    for (Framepoint &fp : fp_vec)
    {
        cv::KeyPoint keypoint;
        if (cam == 0)
        {
            keypoint = fp.keypoint_l.keypoint;
        }
        else if (cam == 1)
        {
            keypoint = fp.keypoint_r.keypoint;
        }
        // Draws a blue cirle on the image
        cv::circle(img_rgb, keypoint.pt, 3, color, CV_FILLED);
    };

    if (cam == 0)
    {
        cv::imshow(left_cam, img_rgb);
    }
    else if (cam == 1)
    {
        cv::imshow(right_cam, img_rgb);
    };
    return;
};

void PoseOptimizer::VisualizeFramepointComparision(FramepointVector fp_vec1, cv::Mat &image_1, FramepointVector fp_vec2, cv::Mat &image_2)
{

    // Convert the images into a RGB image
    // create 8bit color images. IMPORTANT: initialize image otherwise it will result in 32F
    cv::Mat img_rgb1(image_1.size(), CV_8UC3);
    cv::Mat img_rgb2(image_2.size(), CV_8UC3);
    // convert grayscale to color image
    cv::cvtColor(image_1, img_rgb1, CV_GRAY2RGB);
    cv::cvtColor(image_2, img_rgb2, CV_GRAY2RGB);

    for (Framepoint &fp : fp_vec1)
    {
        cv::KeyPoint keypoint;
        keypoint = fp.keypoint_l.keypoint;
        cv::circle(img_rgb1, keypoint.pt, 3, cv::Scalar(0, 0, 255), CV_FILLED);
    };
    for (Framepoint &fp : fp_vec2)
    {
        cv::KeyPoint keypoint;
        keypoint = fp.keypoint_l.keypoint;
        cv::circle(img_rgb2, keypoint.pt, 3, cv::Scalar(255, 0, 0), CV_FILLED);
    };

    cv::imshow(left_cam, img_rgb1);
    cv::imshow(right_cam, img_rgb2);
    cv::waitKey(1000);
    return;
};

void PoseOptimizer::InitializeRANSAC(Frame *current_frame_ptr, float p , float e , int s )
{
    // Initialize RANSAC params
    ransac_params.e = e;
    ransac_params.s = s;
    ransac_params.p = p;
    ransac_params.valid_point_indices.clear();
    ransac_params.max_inliers = 0;
    ransac_params.valid_point_indices.clear();
    ransac_params.inlier_vector_indices.clear();

    // Assign the frame pointer
    this->current_frame_ptr = current_frame_ptr;
    // Number of datapoints
    ransac_params.n = 0;
    for (int i = 0; i < current_frame_ptr->points.size(); i++)
    {
        // TODO : Fix this statement - too shabby
        if (current_frame_ptr->points[i].get()->previous == NULL)
        {
            continue;
        }
        else
        {
            // Adding a valid point for RANSAC-ing later.
            ransac_params.valid_point_indices.push_back(i);
            ransac_params.n++;
        }
    };

    // Calculating T
    assert(ransac_params.p < 1);
    assert(ransac_params.e < 1);
    ransac_params.t = log(1 - p) / (log(1 - pow(1 - e, s)));

    // Initializing the optimizer variables
    H.setZero();
    b.setZero();
    omega.setIdentity();
    T_prev2curr.setIdentity();
    inliers = 0;
    return;
};

int PoseOptimizer::RANSACIterateOnce()
{
    Frame random_frame;
    std::vector<int> random_indices;

    srand(time(NULL)); //< Initializing random seed

    // Selecting 3 random indices
    for (int i = 0; i < 3; i++)
    {
        int random_index = rand() % ransac_params.t;

        // Check for duplicates.
        if (!random_indices.empty())
        {
            while (std::count(random_indices.begin(), random_indices.end(), random_index))
            {
                random_index = rand() & ransac_params.n;
            };
        };

        random_indices.push_back(random_index);
    };

    // Creating vector of randomly selected valid points
    for (int i = 0; i < 3; i++)
    {
        int frame_ptr_index = ransac_params.valid_point_indices[random_indices[i]];
        random_frame.points.push_back(current_frame_ptr->points[frame_ptr_index]);
    };

    // Packaging up the frame
    random_frame.camera_l = current_frame_ptr->camera_l;
    random_frame.camera_r = current_frame_ptr->camera_r;
    random_frame.T_cam2world = current_frame_ptr->T_cam2world;
    random_frame.T_world2cam = current_frame_ptr->T_world2cam;

    // Passing the frame to OptimizeOnce
    OptimizeOnce(&random_frame); // Set the H matrix and b vector.
    Solve();                     // Solve for a new T_prev2curr

    // Checking the number of inliers with this model
    int inliers = 0;
    for (int i = 0; i < ransac_params.n; i++)
    {
        int frame_ptr_index = ransac_params.valid_point_indices[i];
        Framepoint* fp;
        fp = current_frame_ptr->points[frame_ptr_index].get();
        // Now compute the reprojection error
        float error = ComputeError(fp);

        if(error < 10){
            inliers++;
        }
    }

    return inliers;
};
