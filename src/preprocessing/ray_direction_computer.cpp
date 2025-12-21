#include "ray_direction_computer.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>

namespace camera_aware_depth {

RayDirectionComputer::RayDirectionComputer() {
    // Constructor
}

RayDirectionComputer::~RayDirectionComputer() {
    // Destructor
}

Eigen::MatrixXf RayDirectionComputer::computeRayDirections(
    const Eigen::Matrix3f& K,
    int height,
    int width
) {
    // Output: (H*W, 3) matrix where each row is a normalized ray direction
    Eigen::MatrixXf rays(height * width, 3);

    // Compute inverse intrinsics
    Eigen::Matrix3f K_inv = computeInverseIntrinsics(K);

    // Extract intrinsic parameters for faster computation
    float fx = K(0, 0);
    float fy = K(1, 1);
    float cx = K(0, 2);
    float cy = K(1, 2);

    // Precompute inverse values
    float fx_inv = 1.0f / fx;
    float fy_inv = 1.0f / fy;

    // Compute ray direction for each pixel
    int idx = 0;
    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            // Method 1: Using inverse intrinsics (more general)
            // Eigen::Vector3f pixel(static_cast<float>(u), static_cast<float>(v), 1.0f);
            // Eigen::Vector3f ray_unnorm = K_inv * pixel;

            // Method 2: Direct computation (faster)
            float x = (static_cast<float>(u) - cx) * fx_inv;
            float y = (static_cast<float>(v) - cy) * fy_inv;
            float z = 1.0f;

            // Normalize to unit vector
            float norm = std::sqrt(x * x + y * y + z * z);
            rays(idx, 0) = x / norm;
            rays(idx, 1) = y / norm;
            rays(idx, 2) = z / norm;

            ++idx;
        }
    }

    return rays;
}

std::vector<Eigen::MatrixXf> RayDirectionComputer::computeRayDirectionsMaps(
    const Eigen::Matrix3f& K,
    int height,
    int width
) {
    // Output: 3 separate (H, W) matrices for rx, ry, rz components
    std::vector<Eigen::MatrixXf> ray_maps(3);
    ray_maps[0] = Eigen::MatrixXf(height, width); // rx
    ray_maps[1] = Eigen::MatrixXf(height, width); // ry
    ray_maps[2] = Eigen::MatrixXf(height, width); // rz

    // Extract intrinsic parameters
    float fx = K(0, 0);
    float fy = K(1, 1);
    float cx = K(0, 2);
    float cy = K(1, 2);

    // Precompute inverse values
    float fx_inv = 1.0f / fx;
    float fy_inv = 1.0f / fy;

    // Compute ray direction for each pixel
    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            float x = (static_cast<float>(u) - cx) * fx_inv;
            float y = (static_cast<float>(v) - cy) * fy_inv;
            float z = 1.0f;

            // Normalize to unit vector
            float norm = std::sqrt(x * x + y * y + z * z);
            ray_maps[0](v, u) = x / norm;
            ray_maps[1](v, u) = y / norm;
            ray_maps[2](v, u) = z / norm;
        }
    }

    return ray_maps;
}

Eigen::MatrixXf RayDirectionComputer::transformRaysToWorld(
    const Eigen::MatrixXf& rays,
    const Eigen::Matrix4f& pose
) {
    // Extract rotation matrix (3x3) from pose (4x4)
    Eigen::Matrix3f R = pose.block<3, 3>(0, 0);

    // Transform rays: r_world = R * r_camera
    Eigen::MatrixXf rays_world(rays.rows(), 3);

    for (int i = 0; i < rays.rows(); ++i) {
        Eigen::Vector3f ray_camera(rays(i, 0), rays(i, 1), rays(i, 2));
        Eigen::Vector3f ray_world = R * ray_camera;

        // Ray directions should remain normalized after rotation
        // (since R is orthogonal), but normalize anyway for numerical stability
        ray_world.normalize();

        rays_world(i, 0) = ray_world(0);
        rays_world(i, 1) = ray_world(1);
        rays_world(i, 2) = ray_world(2);
    }

    return rays_world;
}

bool RayDirectionComputer::saveRayDirections(
    const Eigen::MatrixXf& rays,
    int height,
    int width,
    const std::string& filename
) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return false;
    }

    // Write header: height and width
    file.write(reinterpret_cast<const char*>(&height), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&width), sizeof(int32_t));

    // Verify dimensions match
    if (rays.rows() != height * width || rays.cols() != 3) {
        std::cerr << "Error: Ray dimensions mismatch. Expected "
                  << height * width << "x3, got "
                  << rays.rows() << "x" << rays.cols() << std::endl;
        file.close();
        return false;
    }

    // Write ray data (row-major order)
    for (int i = 0; i < rays.rows(); ++i) {
        for (int j = 0; j < 3; ++j) {
            float value = rays(i, j);
            file.write(reinterpret_cast<const char*>(&value), sizeof(float));
        }
    }

    file.close();
    std::cout << "Saved ray directions to: " << filename << std::endl;
    std::cout << "  Dimensions: " << height << "x" << width << std::endl;
    std::cout << "  Total rays: " << rays.rows() << std::endl;

    return true;
}

Eigen::MatrixXf RayDirectionComputer::loadRayDirections(
    const std::string& filename,
    int& height,
    int& width
) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file for reading: " + filename);
    }

    // Read header
    file.read(reinterpret_cast<char*>(&height), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&width), sizeof(int32_t));

    // Allocate matrix
    Eigen::MatrixXf rays(height * width, 3);

    // Read ray data
    for (int i = 0; i < height * width; ++i) {
        for (int j = 0; j < 3; ++j) {
            float value;
            file.read(reinterpret_cast<char*>(&value), sizeof(float));
            rays(i, j) = value;
        }
    }

    file.close();
    std::cout << "Loaded ray directions from: " << filename << std::endl;
    std::cout << "  Dimensions: " << height << "x" << width << std::endl;

    return rays;
}

Eigen::Matrix3f RayDirectionComputer::loadIntrinsics(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open intrinsics file: " + filename);
    }

    Eigen::Matrix3f K;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (!(file >> K(i, j))) {
                throw std::runtime_error("Error: Invalid intrinsics file format: " + filename);
            }
        }
    }

    file.close();

    // Validate intrinsic matrix structure
    if (std::abs(K(0, 1)) > 1e-6 || std::abs(K(1, 0)) > 1e-6 ||
        std::abs(K(2, 0)) > 1e-6 || std::abs(K(2, 1)) > 1e-6 ||
        std::abs(K(2, 2) - 1.0f) > 1e-6) {
        std::cerr << "Warning: Intrinsic matrix has unexpected structure" << std::endl;
    }

    std::cout << "Loaded intrinsics from: " << filename << std::endl;
    std::cout << K << std::endl;

    return K;
}

Eigen::Matrix4f RayDirectionComputer::loadPose(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open pose file: " + filename);
    }

    Eigen::Matrix4f pose;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (!(file >> pose(i, j))) {
                throw std::runtime_error("Error: Invalid pose file format: " + filename);
            }
        }
    }

    file.close();

    // Validate pose matrix structure (last row should be [0 0 0 1])
    if (std::abs(pose(3, 0)) > 1e-6 || std::abs(pose(3, 1)) > 1e-6 ||
        std::abs(pose(3, 2)) > 1e-6 || std::abs(pose(3, 3) - 1.0f) > 1e-6) {
        std::cerr << "Warning: Pose matrix has unexpected structure (last row should be [0 0 0 1])" << std::endl;
    }

    // Check if rotation part is orthogonal
    Eigen::Matrix3f R = pose.block<3, 3>(0, 0);
    Eigen::Matrix3f R_T_R = R.transpose() * R;
    if (!R_T_R.isApprox(Eigen::Matrix3f::Identity(), 1e-3f)) {
        std::cerr << "Warning: Rotation matrix is not orthogonal" << std::endl;
    }

    return pose;
}

float RayDirectionComputer::rayDepth(
    const Eigen::Vector3f& ray,
    const Eigen::Vector3f& point
) {
    // Since ray is normalized: depth = dot(point, ray)
    return ray.dot(point);
}

// Private methods

Eigen::Matrix3f RayDirectionComputer::computeInverseIntrinsics(const Eigen::Matrix3f& K) {
    // For standard intrinsic matrix, we can compute inverse analytically
    // K = [fx  0  cx]
    //     [ 0 fy  cy]
    //     [ 0  0   1]
    //
    // K^-1 = [1/fx   0   -cx/fx]
    //        [  0  1/fy  -cy/fy]
    //        [  0    0      1  ]

    float fx = K(0, 0);
    float fy = K(1, 1);
    float cx = K(0, 2);
    float cy = K(1, 2);

    Eigen::Matrix3f K_inv;
    K_inv << 1.0f/fx,      0.0f,  -cx/fx,
                 0.0f,  1.0f/fy,  -cy/fy,
                 0.0f,      0.0f,    1.0f;

    return K_inv;

    // Alternative: Use Eigen's inverse (more general but slower)
    // return K.inverse();
}

Eigen::Vector3f RayDirectionComputer::normalize(const Eigen::Vector3f& vec) {
    float norm = vec.norm();
    if (norm < 1e-8f) {
        std::cerr << "Warning: Attempting to normalize near-zero vector" << std::endl;
        return Eigen::Vector3f(0.0f, 0.0f, 1.0f);
    }
    return vec / norm;
}

} // namespace camera_aware_depth
