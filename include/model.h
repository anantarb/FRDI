#include "constraint.hpp"
#include "hd5_utils.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <math.h>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

using Eigen::MatrixXd;
using Eigen::VectorXd;

class BFMModel {

private:
  void alloc();

  bool load(const std::string &strModelPath);

  unsigned int num_vertices = 47439;
  unsigned int num_pc = 199;
  unsigned int num_faces = 94464;

  unsigned int num_exp_pc = 100;

  VectorXd shape_coef;
  VectorXd shape_mean;
  VectorXd shape_variance;
  MatrixXd shape_pc;

  VectorXd tex_coef;
  VectorXd tex_mean;
  VectorXd tex_variance;
  MatrixXd tex_pc;

  VectorXd exp_coef;
  VectorXd exp_mean;
  VectorXd exp_variance;
  MatrixXd exp_pc;

  Eigen::Matrix<unsigned int, 3, Eigen::Dynamic> faces;
  std::vector<std::vector<unsigned int>> vertices_faces_map;

  VectorXd current_shape;
  VectorXd current_texture;
  VectorXd current_exp;
  VectorXd current_blend_shape;

  VectorXd rotation;
  VectorXd translation;

  VectorXd instrincis;

  std::vector<std::pair<unsigned int, unsigned int>> landmarks_indices;
  VectorXd landmarks_shape_mean;
  MatrixXd landmarks_shape_pc;

  VectorXd landmarks_exp_mean;
  MatrixXd landmarks_exp_pc;

  VectorXd landmark_blend_shape;

public:
  BFMModel(const std::string &ModelPath, const std::string &landmarkPath);

  // setters
  void set_shape_coef(const std::vector<double> coeff);
  void set_texture_coef(const std::vector<double> coeff);
  void set_expression_coef(const std::vector<double> coeff);
  void set_rotation(const double *rot);
  void set_translation(const double *trans);
  void set_instrincis(const double *param);

  // getters
  const unsigned int get_num_vertices() const;
  const unsigned int get_num_pc() const;
  const unsigned int get_num_faces() const;
  const unsigned int get_num_exp_pc() const;

  // getters shape
  const VectorXd get_shape_coef() const;
  const VectorXd get_mean_shape() const;
  const VectorXd get_mean_shape(int vertex_idx) const;
  const VectorXd get_shape_variance() const;
  const MatrixXd get_shape_pc() const;
  const MatrixXd get_shape_pc(int vertex_idx) const;
  const VectorXd get_current_shape() const;
  const VectorXd get_current_shape(int vertex_idx) const;

  // getters texture
  const VectorXd get_tex_coef() const;
  const VectorXd get_mean_tex() const;
  const VectorXd get_mean_tex(int vertex_idx) const;
  const VectorXd get_tex_variance() const;
  const MatrixXd get_tex_pc() const;
  const MatrixXd get_tex_pc(int vertex_idx) const;
  const VectorXd get_current_tex() const;
  const VectorXd get_current_tex(int vertex_idx) const;

  // getters expression
  const VectorXd get_exp_coef() const;
  const VectorXd get_mean_exp() const;
  const VectorXd get_mean_exp(int vertex_idx) const;
  const VectorXd get_exp_variance() const;
  const MatrixXd get_exp_pc() const;
  const MatrixXd get_exp_pc(int vertex_idx) const;
  const VectorXd get_current_exp() const;
  const VectorXd get_current_exp(int vertex_idx) const;

  // other getters
  const VectorXd get_rotation() const;
  const VectorXd get_translation() const;
  const VectorXd get_instrincis() const;

  // other functions
  bool write_model(std::string outfile) const;
  bool write_landmarks(std::string outfile) const;
  void fit_image(std::string LandmarkPath, std::string ImagePath);
};