#include "ceres/cubic_interpolation.h"
#include "opencv2/core/core.hpp"
#include <Eigen/Dense>
#include <array>
#include <ceres/rotation.h>
#include <cmath>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace Constraint {

struct DistortionRegularizationConstraint {
  DistortionRegularizationConstraint(double weight) : weight(weight) {}

  template <typename T>
  bool operator()(const T *const distortion_coefficients, T *residual) const {
    for (unsigned int i = 0; i < 4; ++i) {
      residual[i] = T(weight) * (distortion_coefficients[i] - dist[i]);
    }
    return true;
  };

private:
  const std::vector<double> dist{-0.00348, -7.231, 30.434, -44.4027};
  const double weight = weight;
};

struct RegularizationConstraint {
  RegularizationConstraint(int num_coefficients, const VectorXd &variance,
                           double weight)
      : num_coefficients(num_coefficients), variance(variance), weight(weight) {
  }

  template <typename T>
  bool operator()(const T *const coefficients, T *residual) const {
    for (unsigned int i = 0; i < num_coefficients; ++i) {
      residual[i] = T(weight) * (coefficients[i] / T(sqrt(T(variance(i)))));
    }
    return true;
  };

private:
  const int num_coefficients;
  const VectorXd &variance;
  const double weight = weight;
};

struct LandmarkConstraint {

  LandmarkConstraint(
      const VectorXd &landmarks_shape_mean, const MatrixXd &landmarks_shape_pc,
      const VectorXd &landmarks_exp_mean, const MatrixXd &landmarks_exp_pc,
      const VectorXd &shape_variance, const VectorXd &exp_variance,
      const double observed_landmarks_x, const double observed_landmarks_y,
      unsigned int i, int image_width, int image_height, int num_coefficients)
      : landmarks_shape_mean(landmarks_shape_mean),
        landmarks_shape_pc(landmarks_shape_pc),
        landmarks_exp_mean(landmarks_exp_mean),
        landmarks_exp_pc(landmarks_exp_pc), shape_variance(shape_variance),
        exp_variance(exp_variance), observed_landmarks_x(observed_landmarks_x),
        observed_landmarks_y(observed_landmarks_y), index(i),
        image_width(image_width), image_height(image_height),
        num_coefficients(num_coefficients)

  {}

  template <typename T>
  bool operator()(const T *const cam_extrinsics, const T *const cam_intrinsics,
                  const T *const shape_coefficients,
                  const T *const exp_coefficients,
                  const T *const distortion_coefficients, T *residual) const {

    std::array<T, 3> world_point{T(landmarks_shape_mean(index * 3)),
                                 T(landmarks_shape_mean(index * 3 + 1)),
                                 T(landmarks_shape_mean(index * 3 + 2))};

    // world_point[0] += T(landmarks_exp_mean(index * 3));
    // world_point[1] += T(landmarks_exp_mean(index * 3 + 1));
    // world_point[2] += T(landmarks_exp_mean(index * 3 + 2));

    for (int i = 0; i < num_coefficients; i++) {
      world_point[0] +=
          shape_coefficients[i] * T(landmarks_shape_pc(index * 3, i));
      world_point[1] +=
          shape_coefficients[i] * T(landmarks_shape_pc(index * 3 + 1, i));
      world_point[2] +=
          shape_coefficients[i] * T(landmarks_shape_pc(index * 3 + 2, i));
    }
    // for (int i = 0; i < num_coefficients; i++) {
    //   world_point[0] += exp_coefficients[i] *
    //                     T(landmarks_exp_pc(index * 3, i));
    //   world_point[1] += exp_coefficients[i] *
    //                     T(landmarks_exp_pc(index * 3 + 1, i));
    //   world_point[2] += exp_coefficients[i] *
    //                     T(landmarks_exp_pc(index * 3 + 2, i));
    // }
    const Eigen::Matrix<T, 3, 1> points{world_point[0], world_point[1],
                                        world_point[2]};

    const Eigen::Matrix<T, 3, 3> x_rotation{
        {T(1), T(0), T(0)},
        {T(0), T(cos(cam_extrinsics[0])), T(-sin(cam_extrinsics[0]))},
        {T(0), T(sin(cam_extrinsics[0])), T(cos(cam_extrinsics[0]))}};
    const Eigen::Matrix<T, 3, 3> y_rotation{
        {T(cos(cam_extrinsics[1])), T(0), T(sin(cam_extrinsics[1]))},
        {T(0), T(1), T(0)},
        {T(-sin(cam_extrinsics[1])), T(0), T(cos(cam_extrinsics[1]))}};
    const Eigen::Matrix<T, 3, 3> z_rotation{
        {T(cos(cam_extrinsics[2])), T(-sin(cam_extrinsics[2])), T(0)},
        {T(sin(cam_extrinsics[2])), T(cos(cam_extrinsics[2])), T(0)},
        {T(0), T(0), T(1)}};

    const Eigen::Matrix<T, 3, 3> _rot_mat =
        (z_rotation * y_rotation) * x_rotation;

    // new

    const Eigen::Matrix<T, 3, 1> trans_coord{
        cam_extrinsics[3], cam_extrinsics[4], cam_extrinsics[5]};

    const Eigen::Matrix<T, 3, 1> eye_coord = (_rot_mat * points) + trans_coord;
    const T cx = T(image_width) / T(2.0);
    const T cy = T(image_height) / T(2.0);
    const Eigen::Matrix<T, 3, 1> camera_pos{T(0.0), T(0.0), T(10.0)};

    const Eigen::Matrix<T, 3, 3> reverse_z{{T(1.0), T(0.0), T(0.0)},
                                           {T(0.0), T(1.0), T(0.0)},
                                           {T(0.0), T(0.0), T(-1.0)}};

    const Eigen::Matrix<T, 3, 3> p_matrix{{cam_intrinsics[0], T(0.0), cx},
                                          {T(0.0), cam_intrinsics[1], cy},
                                          {T(0.0), T(0.0), T(1.0)}};

    const Eigen::Matrix<T, 3, 1> temp = (reverse_z * eye_coord) + camera_pos;
    const Eigen::Matrix<T, 3, 1> new_temp{temp(0) / temp(2), temp(1) / temp(2),
                                          temp(2) / temp(2)};
    const Eigen::Matrix<T, 3, 1> norm_coord{new_temp(0), new_temp(1),
                                            new_temp(2)};
    Eigen::Matrix<T, 3, 1> img_coord;

    if (model_distortion == true) {
      const T r2 = sqrt(new_temp(0) * new_temp(0) + new_temp(1) * new_temp(1));
      const T theta = atan(r2);
      const T norm_x = new_temp(0);
      const T norm_y = new_temp(1);

      const T theta_d =
          (theta) *
          (T(1.0) + distortion_coefficients[0] * theta * theta +
           distortion_coefficients[1] * theta * theta * theta * theta +
           distortion_coefficients[2] * theta * theta * theta * theta * theta *
               theta +
           distortion_coefficients[3] * theta * theta * theta * theta * theta *
               theta * theta * theta);

      const T xDistort = ((theta_d) / (r2)) * norm_x;
      const T yDistort = ((theta_d) / (r2)) * norm_y;

      img_coord(0) = cam_intrinsics[0] * (xDistort) + cx;
      img_coord(1) = (cam_intrinsics[1]) * (yDistort) + cy;
      img_coord(2) = new_temp(2);

    } else {
      img_coord = p_matrix * norm_coord;
    }

    const T x = img_coord(0);
    const T y = img_coord(1);
    residual[0] = T(weight) * (x - T(observed_landmarks_x));
    residual[1] = T(weight) * (y - T(observed_landmarks_y));

    return true;
  }

private:
  const VectorXd &landmarks_shape_mean;
  const MatrixXd &landmarks_shape_pc;
  const VectorXd &landmarks_exp_mean;
  const MatrixXd &landmarks_exp_pc;
  const VectorXd &shape_variance;
  const VectorXd &exp_variance;
  const double observed_landmarks_x;
  const double observed_landmarks_y;
  const unsigned int index;
  const int image_width;
  const int image_height;
  const int num_coefficients;
  const bool model_distortion = true;
  const double weight = 1;
};

struct PhotoConstraint {

  PhotoConstraint(
      const VectorXd &shape_mean, const MatrixXd &shape_pc,
      const VectorXd &tex_mean, const MatrixXd &tex_pc,
      const VectorXd &exp_mean, const MatrixXd &exp_pc,
      const VectorXd &shape_variance, const VectorXd &tex_variance,
      const VectorXd &exp_variance, cv::Mat &image, unsigned int i,
      int num_coefficients, const ceres::Grid2D<uchar, 3> &grid,
      const ceres::BiCubicInterpolator<ceres::Grid2D<uchar, 3>> &interpolator,
      const std::vector<std::vector<unsigned int>> &vertices_faces_map,
      const Eigen::Matrix<unsigned int, 3, Eigen::Dynamic> &faces)
      : shape_mean(shape_mean), shape_pc(shape_pc), tex_mean(tex_mean),
        tex_pc(tex_pc), exp_mean(exp_mean), exp_pc(exp_pc),
        shape_variance(shape_variance), tex_variance(tex_variance),
        exp_variance(exp_variance), image(image), index(i),
        num_coefficients(num_coefficients), grid(grid),
        interpolator(interpolator), vertices_faces_map(vertices_faces_map),
        faces(faces)

  {}

  template <typename T>
  bool
  operator()(const T *const cam_extrinsics, const T *const cam_intrinsics,
             const T *const shape_coefficients, const T *const exp_coefficients,
             const T *const tex_coefficients, const T *const light_coefficients,
             const T *const distortion_coefficients, T *residual) const {

    std::array<T, 3> world_point{T(shape_mean(index * 3)),
                                 T(shape_mean(index * 3 + 1)),
                                 T(shape_mean(index * 3 + 2))};

    // world_point[0] += T(exp_mean(index * 3));
    // world_point[1] += T(exp_mean(index * 3 + 1));
    // world_point[2] += T(exp_mean(index * 3 + 2));

    for (int i = 0; i < num_coefficients; i++) {
      world_point[0] += shape_coefficients[i] * T(shape_pc(index * 3, i));
      world_point[1] += shape_coefficients[i] * T(shape_pc(index * 3 + 1, i));
      world_point[2] += shape_coefficients[i] * T(shape_pc(index * 3 + 2, i));
    }

    // for (int i = 0; i < num_coefficients; i++) {
    //   world_point[0] += exp_coefficients[i] *
    //                     T(exp_pc(index * 3, i));
    //   world_point[1] += exp_coefficients[i] *
    //                     T(exp_pc(index * 3 + 1, i));
    //   world_point[2] += exp_coefficients[i] *
    //                     T(exp_pc(index * 3 + 2, i));
    // }

    // computer vertex normal (lightning block)
    std::array<T, 3> vertex_normal{T(0), T(0), T(0)};
    Eigen::Matrix<T, 3, 1> v1;
    Eigen::Matrix<T, 3, 1> v2;
    Eigen::Matrix<T, 3, 1> v3;

    for (int i = 0; i < vertices_faces_map[index].size(); ++i) {
      v1(0) = T(shape_mean(faces(0, vertices_faces_map[index][i]) * 3));
      v1(1) = T(shape_mean(faces(0, vertices_faces_map[index][i]) * 3 + 1));
      v1(2) = T(shape_mean(faces(0, vertices_faces_map[index][i]) * 3 + 2));

      v2(0) = T(shape_mean(faces(1, vertices_faces_map[index][i]) * 3));
      v2(1) = T(shape_mean(faces(1, vertices_faces_map[index][i]) * 3 + 1));
      v2(2) = T(shape_mean(faces(1, vertices_faces_map[index][i]) * 3 + 2));

      v3(0) = T(shape_mean(faces(2, vertices_faces_map[index][i]) * 3));
      v3(1) = T(shape_mean(faces(2, vertices_faces_map[index][i]) * 3 + 1));
      v3(2) = T(shape_mean(faces(2, vertices_faces_map[index][i]) * 3 + 2));

      for (int j = 0; j < num_coefficients; ++j) {
        v1(0) = v1(0) +
                shape_coefficients[j] *
                    T(shape_pc(faces(0, vertices_faces_map[index][i]) * 3, j));
        v1(1) =
            v1(1) +
            shape_coefficients[j] *
                T(shape_pc(faces(0, vertices_faces_map[index][i]) * 3 + 1, j));
        v1(2) =
            v1(2) +
            shape_coefficients[j] *
                T(shape_pc(faces(0, vertices_faces_map[index][i]) * 3 + 2, j));

        v2(0) = v2(0) +
                shape_coefficients[j] *
                    T(shape_pc(faces(1, vertices_faces_map[index][i]) * 3, j));
        v2(1) =
            v2(1) +
            shape_coefficients[j] *
                T(shape_pc(faces(1, vertices_faces_map[index][i]) * 3 + 1, j));
        v2(2) =
            v2(2) +
            shape_coefficients[j] *
                T(shape_pc(faces(1, vertices_faces_map[index][i]) * 3 + 2, j));

        v3(0) = v3(0) +
                shape_coefficients[j] *
                    T(shape_pc(faces(2, vertices_faces_map[index][i]) * 3, j));
        v3(1) =
            v3(1) +
            shape_coefficients[j] *
                T(shape_pc(faces(2, vertices_faces_map[index][i]) * 3 + 1, j));
        v3(2) =
            v3(2) +
            shape_coefficients[j] *
                T(shape_pc(faces(2, vertices_faces_map[index][i]) * 3 + 2, j));
      }
      Eigen::Matrix<T, 3, 1> e1 = v1 - v2;
      Eigen::Matrix<T, 3, 1> e2 = v2 - v3;
      Eigen::Matrix<T, 3, 1> norm = e1.cross(e2).normalized();
      vertex_normal[0] += T(norm(0));
      vertex_normal[1] += T(norm(1));
      vertex_normal[2] += T(norm(2));
    }

    const std::vector<T> init_lit{T(0.8), T(0), T(0), T(0), T(0),
                                  T(0),   T(0), T(0), T(0)};

    const T a0 = T(M_PI);
    const T a1 = T(2) * T(M_PI) / sqrt(3.0);
    const T a2 = T(2) * T(M_PI) / sqrt(8.0);
    const T c0 = T(1) / sqrt(T(4) * T(M_PI));
    const T c1 = sqrt(3.0) / sqrt(T(4) * T(M_PI));
    const T c2 = T(3) * sqrt(5.0) / sqrt(T(12) * T(M_PI));
    const std::vector<T> Y{
        a0 * c0,
        -a1 * c1 * vertex_normal[1],
        a1 * c1 * vertex_normal[2],
        -a1 * c1 * vertex_normal[0],
        a2 * c2 * vertex_normal[0] * vertex_normal[1],
        -a2 * c2 * vertex_normal[1] * vertex_normal[2],
        a2 * c2 * T(0.5) / sqrt(T(3.0)) *
            (T(3) * (vertex_normal[2] - T(1)) * (vertex_normal[2] - T(1))),
        -a2 * c2 * vertex_normal[0] * vertex_normal[2],
        a2 * c2 * T(0.5) *
            ((vertex_normal[0] * vertex_normal[0]) -
             (vertex_normal[1] * vertex_normal[1]))};

    std::array<T, 3> colors{T(0), T(0), T(0)};
    int a = 0;
    for (int i = 0; i < 9; i++) {
      colors[0] += Y[a] * (light_coefficients[i] + init_lit[a]);
      a += 1;
    }
    a = 0;
    for (int i = 9; i < 18; i++) {
      colors[1] += Y[a] * (light_coefficients[i] + init_lit[a]);
      a += 1;
    }
    a = 0;
    for (int i = 18; i < 27; i++) {
      colors[2] += Y[a] * (light_coefficients[i] + init_lit[a]);
      a += 1;
    }

    std::array<T, 3> tex_value{T(tex_mean(index * 3)),
                               T(tex_mean(index * 3 + 1)),
                               T(tex_mean(index * 3 + 2))};
    for (int i = 0; i < num_coefficients; i++) {
      tex_value[0] += tex_coefficients[i] * T(tex_pc(index * 3, i));
      tex_value[1] += tex_coefficients[i] * T(tex_pc(index * 3 + 1, i));
      tex_value[2] += tex_coefficients[i] * T(tex_pc(index * 3 + 2, i));
    }

    const Eigen::Matrix<T, 3, 1> points{world_point[0], world_point[1],
                                        world_point[2]};

    std::array<T, 3> face_color{colors[0] * tex_value[0],
                                colors[1] * tex_value[1],
                                colors[2] * tex_value[2]};

    const Eigen::Matrix<T, 3, 3> x_rotation{
        {T(1), T(0), T(0)},
        {T(0), T(cos(cam_extrinsics[0])), T(-sin(cam_extrinsics[0]))},
        {T(0), T(sin(cam_extrinsics[0])), T(cos(cam_extrinsics[0]))}};
    const Eigen::Matrix<T, 3, 3> y_rotation{
        {T(cos(cam_extrinsics[1])), T(0), T(sin(cam_extrinsics[1]))},
        {T(0), T(1), T(0)},
        {T(-sin(cam_extrinsics[1])), T(0), T(cos(cam_extrinsics[1]))}};
    const Eigen::Matrix<T, 3, 3> z_rotation{
        {T(cos(cam_extrinsics[2])), T(-sin(cam_extrinsics[2])), T(0)},
        {T(sin(cam_extrinsics[2])), T(cos(cam_extrinsics[2])), T(0)},
        {T(0), T(0), T(1)}};

    const Eigen::Matrix<T, 3, 3> _rot_mat =
        z_rotation * y_rotation * x_rotation;

    // new

    const Eigen::Matrix<T, 3, 1> trans_coord{
        cam_extrinsics[3], cam_extrinsics[4], cam_extrinsics[5]};

    const Eigen::Matrix<T, 3, 1> eye_coord = (_rot_mat * points) + trans_coord;

    // const T focal_x = cam_intrinsics[0];
    // const T focal_y = cam_intrinsics[1];
    const T cx = T(image.cols) / T(2.0);
    const T cy = T(image.rows) / T(2.0);
    const Eigen::Matrix<T, 3, 1> camera_pos{T(0.0), T(0.0), T(10.0)};

    const Eigen::Matrix<T, 3, 3> reverse_z{{T(1.0), T(0.0), T(0.0)},
                                           {T(0.0), T(1.0), T(0.0)},
                                           {T(0.0), T(0.0), T(-1.0)}};

    const Eigen::Matrix<T, 3, 3> p_matrix{{cam_intrinsics[0], T(0.0), cx},
                                          {T(0.0), cam_intrinsics[1], cy},
                                          {T(0.0), T(0.0), T(1.0)}};

    const Eigen::Matrix<T, 3, 1> temp = (reverse_z * eye_coord) + camera_pos;

    const Eigen::Matrix<T, 3, 1> new_temp{temp(0) / temp(2), temp(1) / temp(2),
                                          temp(2) / temp(2)};
    const Eigen::Matrix<T, 3, 1> norm_coord{new_temp(0), new_temp(1),
                                            new_temp(2)};
    Eigen::Matrix<T, 3, 1> img_coord;

    if (model_distortion == true) {
      const T r2 = sqrt(new_temp(0) * new_temp(0) + new_temp(1) * new_temp(1));
      const T theta = atan(r2);
      const T norm_x = new_temp(0);
      const T norm_y = new_temp(1);

      const T theta_d =
          (theta) *
          (T(1.0) + distortion_coefficients[0] * theta * theta +
           distortion_coefficients[1] * theta * theta * theta * theta +
           distortion_coefficients[2] * theta * theta * theta * theta * theta *
               theta +
           distortion_coefficients[3] * theta * theta * theta * theta * theta *
               theta * theta * theta);

      const T xDistort = ((theta_d) / (r2)) * norm_x;
      const T yDistort = ((theta_d) / (r2)) * norm_y;

      img_coord(0) = cam_intrinsics[0] * (xDistort) + cx;
      img_coord(1) = (cam_intrinsics[1]) * (yDistort) + cy;
      img_coord(2) = new_temp(2);

    } else {
      img_coord = p_matrix * norm_coord;
    }

    const T x = img_coord(0);
    const T y = img_coord(1);

    if (y < T(0) || y >= T(image.rows) || x < T(0) || x >= T(image.cols)) {
      residual[0] = T(0.0);
      residual[1] = T(0.0);
      residual[2] = T(0.0);

      return true;
    } else {

      T observed_colour[3];
      interpolator.Evaluate(y, x, &observed_colour[0]);
      residual[0] =
          T(weight) * (face_color[0] - (T(observed_colour[2]) / T(255.0)));
      residual[1] =
          T(weight) * (face_color[1] - (T(observed_colour[1]) / T(255.0)));
      residual[2] =
          T(weight) * (face_color[2] - (T(observed_colour[0]) / T(255.0)));
    }

    return true;
  }

private:
  const VectorXd &shape_mean;
  const MatrixXd &shape_pc;
  const VectorXd &tex_mean;
  const MatrixXd &tex_pc;
  const VectorXd &exp_mean;
  const MatrixXd &exp_pc;
  const VectorXd &shape_variance;
  const VectorXd &tex_variance;
  const VectorXd &exp_variance;
  const cv::Mat &image;
  const unsigned int index;
  const int num_coefficients;

  const ceres::Grid2D<uchar, 3> &grid;
  const ceres::BiCubicInterpolator<ceres::Grid2D<uchar, 3>> &interpolator;

  const std::vector<std::vector<unsigned int>> &vertices_faces_map;
  const Eigen::Matrix<unsigned int, 3, Eigen::Dynamic> &faces;
  const bool model_distortion = true;
  const double weight = 1;
};
} // namespace Constraint