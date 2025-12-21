#ifndef RAY_DIRECTION_COMPUTER_H
#define RAY_DIRECTION_COMPUTER_H

#include <Eigen/Dense>
#include <Eigen/Core>
#include <string>
#include <vector>
#include <memory>

namespace camera_aware_depth {

/**
 * @brief Ray Direction Computer
 *
 * Computes per-pixel ray directions from camera intrinsics.
 * Ray directions are normalized 3D vectors pointing from the camera center
 * through each pixel in the image plane.
 *
 * Mathematical Formulation:
 *   For pixel (u, v):
 *   1. Normalize: [x, y, 1]^T = K^-1 * [u, v, 1]^T
 *   2. Compute ray: r = normalize([x, y, 1]^T)
 *
 * Where K is the camera intrinsic matrix:
 *   K = [fx  0  cx]
 *       [ 0 fy  cy]
 *       [ 0  0   1]
 */
class RayDirectionComputer {
public:
    /**
     * @brief Construct a new Ray Direction Computer object
     */
    RayDirectionComputer();

    /**
     * @brief Destroy the Ray Direction Computer object
     */
    ~RayDirectionComputer();

    /**
     * @brief Compute ray directions for all pixels in an image
     *
     * @param K Camera intrinsic matrix (3x3)
     * @param height Image height in pixels
     * @param width Image width in pixels
     * @return Eigen::MatrixXf Ray directions (H*W, 3) where each row is a normalized ray
     *
     * Output format: Each row contains [rx, ry, rz] where ||r|| = 1
     */
    Eigen::MatrixXf computeRayDirections(
        const Eigen::Matrix3f& K,
        int height,
        int width
    );

    /**
     * @brief Compute ray directions and reshape to image dimensions
     *
     * @param K Camera intrinsic matrix (3x3)
     * @param height Image height in pixels
     * @param width Image width in pixels
     * @return std::vector<Eigen::MatrixXf> Vector of 3 matrices (rx, ry, rz) each of size (H, W)
     */
    std::vector<Eigen::MatrixXf> computeRayDirectionsMaps(
        const Eigen::Matrix3f& K,
        int height,
        int width
    );

    /**
     * @brief Transform ray directions from camera space to world space
     *
     * @param rays Input ray directions (H*W, 3) or (H, W, 3)
     * @param pose Camera extrinsic matrix (4x4) - camera-to-world transformation
     * @return Eigen::MatrixXf Transformed ray directions in world coordinates
     *
     * Transformation: r_world = R * r_camera
     * where R is the 3x3 rotation part of the pose matrix
     */
    Eigen::MatrixXf transformRaysToWorld(
        const Eigen::MatrixXf& rays,
        const Eigen::Matrix4f& pose
    );

    /**
     * @brief Save ray directions to binary file
     *
     * @param rays Ray directions matrix (H*W, 3)
     * @param height Image height
     * @param width Image width
     * @param filename Output file path (*.bin)
     * @return true if save successful
     * @return false if save failed
     *
     * File format:
     *   - 4 bytes: height (int32)
     *   - 4 bytes: width (int32)
     *   - H*W*3*4 bytes: ray data (float32, row-major)
     */
    bool saveRayDirections(
        const Eigen::MatrixXf& rays,
        int height,
        int width,
        const std::string& filename
    );

    /**
     * @brief Load ray directions from binary file
     *
     * @param filename Input file path (*.bin)
     * @param height Output image height
     * @param width Output image width
     * @return Eigen::MatrixXf Ray directions matrix (H*W, 3)
     */
    Eigen::MatrixXf loadRayDirections(
        const std::string& filename,
        int& height,
        int& width
    );

    /**
     * @brief Load camera intrinsics from file
     *
     * @param filename Path to intrinsic.txt file
     * @return Eigen::Matrix3f Camera intrinsic matrix (3x3)
     *
     * Expected format (3 rows, 3 columns):
     *   fx  0  cx
     *    0 fy  cy
     *    0  0   1
     */
    static Eigen::Matrix3f loadIntrinsics(const std::string& filename);

    /**
     * @brief Load camera pose from file
     *
     * @param filename Path to pose.txt file
     * @return Eigen::Matrix4f Camera extrinsic matrix (4x4)
     *
     * Expected format (4 rows, 4 columns):
     *   r11 r12 r13 tx
     *   r21 r22 r23 ty
     *   r31 r32 r33 tz
     *     0   0   0  1
     */
    static Eigen::Matrix4f loadPose(const std::string& filename);

    /**
     * @brief Compute depth from ray direction and 3D point
     *
     * @param ray Normalized ray direction (3,)
     * @param point 3D point in camera coordinates (3,)
     * @return float Depth value (scalar distance along ray)
     *
     * Formula: depth = dot(point, ray) / ||ray||^2
     * Since rays are normalized: depth = dot(point, ray)
     */
    static float rayDepth(
        const Eigen::Vector3f& ray,
        const Eigen::Vector3f& point
    );

private:
    /**
     * @brief Compute inverse of intrinsic matrix
     *
     * @param K Input intrinsic matrix
     * @return Eigen::Matrix3f Inverse intrinsic matrix
     */
    Eigen::Matrix3f computeInverseIntrinsics(const Eigen::Matrix3f& K);

    /**
     * @brief Normalize a 3D vector to unit length
     *
     * @param vec Input vector
     * @return Eigen::Vector3f Normalized vector
     */
    Eigen::Vector3f normalize(const Eigen::Vector3f& vec);
};

} // namespace camera_aware_depth

#endif // RAY_DIRECTION_COMPUTER_H
