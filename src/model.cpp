#include "../include/model.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

BFMModel::BFMModel(const std::string &ModelPath,
                   const std::string &landmarkPath) {

  // pre-checks
  if (!fs::exists(ModelPath)) {
    std::cout << "Can't load the Model.";
    return;
  }

  if (!fs::exists(landmarkPath)) {
    std::cout << "Can't load the landmark file.";
    return;
  } else {

    // load the 3D landmarks of the model
    std::ifstream inFile;
    inFile.open(landmarkPath, std::ios::in);
    unsigned int j = 0;
    unsigned int index;
    while (inFile >> index) {
      landmarks_indices.push_back(std::make_pair(j, index));
      j++;
    }
    inFile.close();
  }

  // allocate the memory for the model
  this->alloc();
  // load the model into allocated vectors
  this->load(ModelPath);

  // init values
  current_shape = shape_mean;
  current_texture = tex_mean;
  current_exp = exp_mean;
  current_blend_shape = current_shape; //+ current_exp;
  for (unsigned int j = 0; j < shape_coef.rows(); ++j) {
    shape_coef(j) = 0;
  }
  for (unsigned int j = 0; j < exp_coef.rows(); ++j) {
    exp_coef(j) = 0;
  }

  // identify landmark vertices
  for (unsigned int j = 0; j < landmarks_indices.size(); ++j) {
    landmarks_shape_mean(j * 3) = shape_mean(landmarks_indices[j].second * 3);
    landmarks_shape_mean(j * 3 + 1) =
        shape_mean(landmarks_indices[j].second * 3 + 1);
    landmarks_shape_mean(j * 3 + 2) =
        shape_mean(landmarks_indices[j].second * 3 + 2);

    landmarks_exp_mean(j * 3) = exp_mean(landmarks_indices[j].second * 3);
    landmarks_exp_mean(j * 3 + 1) =
        exp_mean(landmarks_indices[j].second * 3 + 1);
    landmarks_exp_mean(j * 3 + 2) =
        exp_mean(landmarks_indices[j].second * 3 + 2);

    for (unsigned int i = 0; i < num_pc; ++i) {
      landmarks_shape_pc(j * 3, i) =
          shape_pc(landmarks_indices[j].second * 3, i);
      landmarks_shape_pc(j * 3 + 1, i) =
          shape_pc(landmarks_indices[j].second * 3 + 1, i);
      landmarks_shape_pc(j * 3 + 2, i) =
          shape_pc(landmarks_indices[j].second * 3 + 2, i);
    }

    for (unsigned int i = 0; i < num_exp_pc; ++i) {
      landmarks_exp_pc(j * 3, i) = exp_pc(landmarks_indices[j].second * 3, i);
      landmarks_exp_pc(j * 3 + 1, i) =
          exp_pc(landmarks_indices[j].second * 3 + 1, i);
      landmarks_exp_pc(j * 3 + 2, i) =
          exp_pc(landmarks_indices[j].second * 3 + 2, i);
    }
  }

  landmark_blend_shape = landmarks_shape_mean + landmarks_exp_mean;

  // map the vertices to faces
  std::vector<unsigned int> temp;
  for (unsigned int i = 0; i < num_vertices; ++i) {
    temp.resize(0);
    for (unsigned int j = 0; j < num_faces; ++j) {
      if (faces(0, j) == i || faces(1, j) == i || faces(2, j) == i) {
        temp.push_back(j);
      }
    }
    vertices_faces_map.push_back(temp);
  }
}

void BFMModel::alloc() {
  std::cout << "Allocating Memory for the Model." << std::endl;
  shape_coef.resize(num_pc);
  shape_mean.resize(num_vertices * 3);
  shape_variance.resize(num_pc);
  shape_pc.resize(num_vertices * 3, num_pc);

  tex_coef.resize(num_pc);
  tex_mean.resize(num_vertices * 3);
  tex_variance.resize(num_pc);
  tex_pc.resize(num_vertices * 3, num_pc);

  exp_coef.resize(num_exp_pc);
  exp_mean.resize(num_vertices * 3);
  exp_variance.resize(num_exp_pc);
  exp_pc.resize(num_vertices * 3, num_exp_pc);

  faces.resize(3, num_faces);

  current_shape.resize(num_vertices * 3);
  current_texture.resize(num_vertices * 3);
  current_exp.resize(num_vertices * 3);
  current_blend_shape.resize(num_vertices * 3);

  rotation.resize(3);
  rotation.setZero(3);
  translation.resize(3);
  translation.setZero(3);
  instrincis.resize(4);
  instrincis.setZero(4);

  landmarks_shape_mean.resize(landmarks_indices.size() * 3);
  landmarks_shape_pc.resize(landmarks_indices.size() * 3, num_pc);
  landmarks_exp_mean.resize(landmarks_indices.size() * 3);
  landmarks_exp_pc.resize(landmarks_indices.size() * 3, num_exp_pc);
  landmark_blend_shape.resize(landmarks_indices.size() * 3);
}

bool BFMModel::load(const std::string &strModelPath) {
  std::cout << "Loading the Model into Memory." << std::endl;
  std::unique_ptr<double[]> _shape_mean(new double[num_vertices * 3]);
  std::unique_ptr<double[]> _shape_variance(new double[num_pc]);
  std::unique_ptr<double[]> _shape_pc(new double[num_vertices * 3 * num_pc]);
  std::unique_ptr<double[]> _tex_mean(new double[num_vertices * 3]);
  std::unique_ptr<double[]> _tex_variance(new double[num_pc]);
  std::unique_ptr<double[]> _tex_pc(new double[num_vertices * 3 * num_pc]);

  std::unique_ptr<double[]> _exp_mean(new double[num_vertices * 3]);
  std::unique_ptr<double[]> _exp_variance(new double[num_exp_pc]);
  std::unique_ptr<double[]> _exp_pc(new double[num_vertices * 3 * num_exp_pc]);

  std::unique_ptr<unsigned int[]> _faces(new unsigned int[3 * num_faces]);

  hid_t file = H5Fopen(strModelPath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  hd5_utils::LoadH5Model(file, "shape/model/mean", _shape_mean, shape_mean,
                         H5T_NATIVE_DOUBLE);
  hd5_utils::LoadH5Model(file, "shape/model/pcaVariance", _shape_variance,
                         shape_variance, H5T_NATIVE_DOUBLE);
  hd5_utils::LoadH5Model(file, "shape/model/pcaBasis", _shape_pc, shape_pc,
                         H5T_NATIVE_DOUBLE);

  hd5_utils::LoadH5Model(file, "color/model/mean", _tex_mean, tex_mean,
                         H5T_NATIVE_DOUBLE);
  hd5_utils::LoadH5Model(file, "color/model/pcaVariance", _tex_variance,
                         tex_variance, H5T_NATIVE_DOUBLE);
  hd5_utils::LoadH5Model(file, "color/model/pcaBasis", _tex_pc, tex_pc,
                         H5T_NATIVE_DOUBLE);

  hd5_utils::LoadH5Model(file, "expression/model/mean", _exp_mean, exp_mean,
                         H5T_NATIVE_DOUBLE);
  hd5_utils::LoadH5Model(file, "expression/model/pcaVariance", _exp_variance,
                         exp_variance, H5T_NATIVE_DOUBLE);
  hd5_utils::LoadH5Model(file, "expression/model/pcaBasis", _exp_pc, exp_pc,
                         H5T_NATIVE_DOUBLE);

  hd5_utils::LoadH5Model(file, "shape/representer/cells", _faces, faces,
                         H5T_NATIVE_INT);

  std::cout << "Finished Loading the Model into Memory." << std::endl;

  return true;
}

bool BFMModel::write_model(std::string outfile) const {
  std::cout << "Writing the model to the OFF file" << std::endl;
  std::ofstream outFile(outfile);
  if (!outFile.is_open())
    return false;
  outFile << "COFF" << std::endl;
  outFile << "# numVertices numFaces numEdges" << std::endl;
  outFile << num_vertices << " " << num_faces << " 0" << std::endl;
  for (unsigned int i = 0; i < num_vertices; i++) {

    double x = current_blend_shape(i * 3);
    double y = current_blend_shape(i * 3 + 1);
    double z = current_blend_shape(i * 3 + 2);
    double r = current_texture(i * 3);
    double g = current_texture(i * 3 + 1);
    double b = current_texture(i * 3 + 2);
    outFile << x << " " << y << " " << z << " " << r << " " << g << " " << b
            << " " << std::endl;
  }

  for (unsigned int i = 0; i < num_faces; ++i) {
    int x = faces(0, i);
    int y = faces(1, i);
    int z = faces(2, i);
    outFile << 3 << " " << x << " " << y << " " << z << std::endl;
  }
  outFile.close();

  return true;
}

bool BFMModel::write_landmarks(std::string outfile) const {
  std::cout << "Writing the landmarks to the OFF file" << std::endl;
  std::ofstream outFile(outfile);
  if (!outFile.is_open())
    return false;
  outFile << "COFF" << std::endl;
  outFile << "# numVertices numFaces numEdges" << std::endl;
  outFile << landmarks_indices.size() << " "
          << "0"
          << " "
          << "0" << std::endl;
  for (unsigned int i = 0; i < landmarks_indices.size(); i++) {

    double x = landmark_blend_shape(i * 3);
    double y = landmark_blend_shape(i * 3 + 1);
    double z = landmark_blend_shape(i * 3 + 2);
    outFile << x << " " << y << " " << z << " "
            << "0"
            << " "
            << "0"
            << " "
            << "0"
            << " " << std::endl;
  }
  outFile.close();

  return true;
}

// setters
void BFMModel::set_shape_coef(const std::vector<double> coeff) {
  for (unsigned int i = 0; i < coeff.size(); ++i) {
    shape_coef(i) = coeff[i];
  }

  current_shape = shape_mean + shape_pc * shape_coef;
  current_blend_shape = current_shape;
}

void BFMModel::set_texture_coef(const std::vector<double> coeff) {
  for (unsigned int i = 0; i < coeff.size(); ++i) {
    tex_coef(i) = coeff[i];
  }

  current_texture = tex_mean + tex_pc * tex_coef;
}

void BFMModel::set_expression_coef(const std::vector<double> coeff) {
  for (unsigned int i = 0; i < coeff.size(); ++i) {
    exp_coef(i) = coeff[i];
  }

  current_exp = exp_mean + exp_pc * exp_coef;
  current_blend_shape = current_shape + current_exp;
}

void BFMModel::set_rotation(const double *rot) {
  rotation(0) = rot[0];
  rotation(1) = rot[1];
  rotation(2) = rot[2];
}

void BFMModel::set_translation(const double *trans) {
  translation(0) = trans[0];
  translation(1) = trans[1];
  translation(2) = trans[2];
}

void BFMModel::set_instrincis(const double *param) {
  instrincis(0) = param[0];
  instrincis(1) = param[1];
  instrincis(2) = param[2];
}

// getters shape
const VectorXd BFMModel::get_shape_coef() const { return shape_coef; }

const VectorXd BFMModel::get_mean_shape() const { return shape_mean; }

const VectorXd BFMModel::get_mean_shape(int vertex_idx) const {
  return shape_mean(Eigen::seq(3 * vertex_idx, 3 * vertex_idx + 2));
}

const VectorXd BFMModel::get_shape_variance() const { return shape_variance; }

const MatrixXd BFMModel::get_shape_pc() const { return shape_pc; }

const MatrixXd BFMModel::get_shape_pc(int vertex_idx) const {
  return shape_pc(Eigen::seq(3 * vertex_idx, 3 * vertex_idx + 2),
                  Eigen::seq(0, num_pc));
}

const VectorXd BFMModel::get_current_shape() const { return current_shape; }

const VectorXd BFMModel::get_current_shape(int vertex_idx) const {
  return current_shape(Eigen::seq(3 * vertex_idx, 3 * vertex_idx + 2));
}

// getters tex
const VectorXd BFMModel::get_tex_coef() const { return tex_coef; }

const VectorXd BFMModel::get_mean_tex() const { return tex_mean; }

const VectorXd BFMModel::get_mean_tex(int vertex_idx) const {
  return tex_mean(Eigen::seq(3 * vertex_idx, 3 * vertex_idx + 2));
}

const VectorXd BFMModel::get_tex_variance() const { return tex_variance; }

const MatrixXd BFMModel::get_tex_pc() const { return tex_pc; }

const MatrixXd BFMModel::get_tex_pc(int vertex_idx) const {
  return tex_pc(Eigen::seq(3 * vertex_idx, 3 * vertex_idx + 2),
                Eigen::seq(0, num_pc));
}

const VectorXd BFMModel::get_current_tex() const { return current_texture; }

const VectorXd BFMModel::get_current_tex(int vertex_idx) const {
  return current_texture(Eigen::seq(3 * vertex_idx, 3 * vertex_idx + 2));
}

// getters exp
const VectorXd BFMModel::get_exp_coef() const { return exp_coef; }

const VectorXd BFMModel::get_mean_exp() const { return exp_mean; }

const VectorXd BFMModel::get_mean_exp(int vertex_idx) const {
  return exp_mean(Eigen::seq(3 * vertex_idx, 3 * vertex_idx + 2));
}

const VectorXd BFMModel::get_exp_variance() const { return exp_variance; }

const MatrixXd BFMModel::get_exp_pc() const { return exp_pc; }

const MatrixXd BFMModel::get_exp_pc(int vertex_idx) const {
  return exp_pc(Eigen::seq(3 * vertex_idx, 3 * vertex_idx + 2),
                Eigen::seq(0, num_pc));
}

const VectorXd BFMModel::get_current_exp() const { return current_exp; }

const VectorXd BFMModel::get_current_exp(int vertex_idx) const {
  return current_exp(Eigen::seq(3 * vertex_idx, 3 * vertex_idx + 2));
}

const unsigned int BFMModel::get_num_vertices() const { return num_vertices; }

// other getters
const unsigned int BFMModel::get_num_pc() const { return num_pc; }

const unsigned int BFMModel::get_num_faces() const { return num_faces; }

const unsigned int BFMModel::get_num_exp_pc() const { return num_exp_pc; }

const VectorXd BFMModel::get_rotation() const { return rotation; }

const VectorXd BFMModel::get_translation() const { return translation; }

const VectorXd BFMModel::get_instrincis() const { return instrincis; }

void BFMModel::fit_image(std::string LandmarkPath, std::string ImagePath) {

  // loading necessary files and image
  if (!fs::exists(LandmarkPath)) {
    std::cout << "Can't load the Landmarks.";
    return;
  }
  if (!fs::exists(ImagePath)) {
    std::cout << "Can't load the Image.";
    return;
  }
  cv::Mat image = cv::imread(ImagePath);

  double i, j;
  std::ifstream inFile;
  std::vector<std::pair<double, double>> observed_landmarks;
  inFile.open(LandmarkPath, std::ios::in);
  while (inFile >> i >> j) {
    observed_landmarks.push_back(std::make_pair(i, j));
  }
  inFile.close();

  // start setting up parameters for fitting landmarks
  std::cout << "Fitting Model to the landmarks." << std::endl;
  std::vector<double> cam_extrinsics;
  std::vector<double> shape_coefficients;
  std::vector<double> tex_coefficients;
  std::vector<double> exp_coefficients;
  std::vector<double> cam_intrinsics;
  std::vector<double> light_coefficients;
  //std::vector<double> distortion_coefficients{-0.00348, -7.231, 30.434, -44.4027};
  std::vector<double> distortion_coefficients;
  const int num_coef = 20;
  const int num_tex_coef = 20;

  cam_extrinsics.resize(6);
  cam_intrinsics.resize(2);
  cam_intrinsics[0] = 1015.0;
  cam_intrinsics[1] = 1015.0;
  shape_coefficients.resize(num_coef);
  exp_coefficients.resize(num_coef);
  tex_coefficients.resize(num_tex_coef);
  light_coefficients.resize(27);
  distortion_coefficients.resize(4);

  //first optmization step
  ceres::Problem problem;

  // ceres residual block
  for (unsigned int i = 0; i < observed_landmarks.size(); ++i) {
    ceres::CostFunction *cost_function =
        new ceres::AutoDiffCostFunction<Constraint::LandmarkConstraint, 2, 6, 2,
                                        num_coef, num_coef, 4>(
            new Constraint::LandmarkConstraint(
                landmarks_shape_mean, landmarks_shape_pc, landmarks_exp_mean,
                landmarks_exp_pc, shape_variance, exp_variance,
                observed_landmarks[i].first, observed_landmarks[i].second, i,
                image.cols, image.rows, num_coef));
    problem.AddResidualBlock(cost_function, nullptr, &cam_extrinsics[0],
                             &cam_intrinsics[0], &shape_coefficients[0],
                             &exp_coefficients[0], &distortion_coefficients[0]);
  }

  ceres::CostFunction *dist_prior =
      new ceres::AutoDiffCostFunction<Constraint::DistortionRegularizationConstraint,
                                      4, 4>(new Constraint::DistortionRegularizationConstraint(1.0));


  problem.AddResidualBlock(dist_prior, nullptr, &distortion_coefficients[0]);
  problem.SetParameterBlockConstant(&exp_coefficients[0]);
  problem.SetParameterBlockConstant(&shape_coefficients[0]);
  //problem.SetParameterBlockConstant(&distortion_coefficients[0]);

  ceres::Solver::Options options;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.use_nonmonotonic_steps = false;
  options.linear_solver_type = ceres::DENSE_QR;
  options.num_threads = 8;
  options.max_num_iterations = 100;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;

  // now fit using the whole image
  // second optmization step
  ceres::Problem new_problem;
  ceres::Grid2D<uchar, 3> grid(image.ptr(0), 0, image.rows, 0, image.cols);
  ceres::BiCubicInterpolator<ceres::Grid2D<uchar, 3>> interpolator(grid);

  std::cout << "Fitting Model to the Image." << std::endl;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, num_vertices);

  for (unsigned int i = 0; i < 10000; ++i) {
    unsigned int index = distrib(gen);
    ceres::CostFunction *cost_function =
        new ceres::AutoDiffCostFunction<Constraint::PhotoConstraint, 3, 6, 2,
                                        num_coef, num_coef, num_tex_coef, 27,
                                        4>(new Constraint::PhotoConstraint(
            shape_mean, shape_pc, tex_mean, tex_pc, exp_mean, exp_pc,
            shape_variance, tex_variance, exp_variance, image, index, num_coef,
            grid, interpolator, vertices_faces_map, faces));
    new_problem.AddResidualBlock(
        cost_function, nullptr, &cam_extrinsics[0], &cam_intrinsics[0],
        &shape_coefficients[0], &exp_coefficients[0], &tex_coefficients[0],
        &light_coefficients[0], &distortion_coefficients[0]);
  }

  // set other constraints
  ceres::CostFunction *shape_prior =
      new ceres::AutoDiffCostFunction<Constraint::RegularizationConstraint,
                                      num_coef, num_coef>(
          new Constraint::RegularizationConstraint(num_coef, shape_variance, 1));

  ceres::CostFunction *tex_prior =
      new ceres::AutoDiffCostFunction<Constraint::RegularizationConstraint,
                                      num_tex_coef, num_tex_coef>(
          new Constraint::RegularizationConstraint(num_tex_coef, tex_variance, 1));

  ceres::CostFunction *exp_prior =
      new ceres::AutoDiffCostFunction<Constraint::RegularizationConstraint,
                                      num_coef, num_coef>(
          new Constraint::RegularizationConstraint(num_coef, exp_variance, 1));

  ceres::CostFunction *dist_prior_one =
      new ceres::AutoDiffCostFunction<Constraint::DistortionRegularizationConstraint,
                                      4, 4>(new Constraint::DistortionRegularizationConstraint(1.0));

  new_problem.AddResidualBlock(dist_prior_one, nullptr, &distortion_coefficients[0]);

  new_problem.AddResidualBlock(tex_prior, nullptr, &tex_coefficients[0]);
  new_problem.AddResidualBlock(shape_prior, nullptr, &shape_coefficients[0]);
  new_problem.AddResidualBlock(exp_prior, nullptr, &exp_coefficients[0]);

  new_problem.SetParameterBlockConstant(&exp_coefficients[0]);
  ceres::Solve(options, &new_problem, &summary);
  std::cout << summary.BriefReport() << std::endl;
  set_shape_coef(shape_coefficients);
  set_texture_coef(tex_coefficients);
  write_model("face.off");


  // third optmization step
  ceres::Problem new_problem_two;
  const int new_num_coeff = 50;
  const int new_num_tex_coeff = 50;
  shape_coefficients.resize(new_num_coeff);
  tex_coefficients.resize(new_num_tex_coeff);
  for (unsigned int i = 0; i < 30000; ++i) {
    unsigned int index = distrib(gen);
    ceres::CostFunction *cost_function_two =
        new ceres::AutoDiffCostFunction<Constraint::PhotoConstraint, 3, 6, 2,
                                        new_num_coeff, new_num_coeff, new_num_tex_coeff, 27,
                                        4>(new Constraint::PhotoConstraint(
            shape_mean, shape_pc, tex_mean, tex_pc, exp_mean, exp_pc,
            shape_variance, tex_variance, exp_variance, image, index, new_num_coeff,
            grid, interpolator, vertices_faces_map, faces));
    new_problem_two.AddResidualBlock(
        cost_function_two, nullptr, &cam_extrinsics[0], &cam_intrinsics[0],
        &shape_coefficients[0], &exp_coefficients[0], &tex_coefficients[0],
        &light_coefficients[0], &distortion_coefficients[0]);
  }

  // set other constraints
  ceres::CostFunction *shape_prior_two =
      new ceres::AutoDiffCostFunction<Constraint::RegularizationConstraint,
                                      new_num_coeff, new_num_coeff>(
          new Constraint::RegularizationConstraint(new_num_coeff, shape_variance, 1.0));

  ceres::CostFunction *tex_prior_two =
      new ceres::AutoDiffCostFunction<Constraint::RegularizationConstraint,
                                      new_num_tex_coeff, new_num_tex_coeff>(
          new Constraint::RegularizationConstraint(new_num_tex_coeff, tex_variance, 1.0));

  ceres::CostFunction *exp_prior_two =
      new ceres::AutoDiffCostFunction<Constraint::RegularizationConstraint,
                                      new_num_coeff, new_num_coeff>(
          new Constraint::RegularizationConstraint(new_num_coeff, exp_variance, 1.0));

  ceres::CostFunction *dist_prior_two =
      new ceres::AutoDiffCostFunction<Constraint::DistortionRegularizationConstraint,
                                      4, 4>(new Constraint::DistortionRegularizationConstraint(1.0));
  
  new_problem_two.AddResidualBlock(dist_prior_two, nullptr, &distortion_coefficients[0]);

  new_problem_two.AddResidualBlock(tex_prior_two, nullptr, &tex_coefficients[0]);
  new_problem_two.AddResidualBlock(shape_prior_two, nullptr, &shape_coefficients[0]);
  new_problem_two.AddResidualBlock(exp_prior_two, nullptr, &exp_coefficients[0]);

  new_problem_two.SetParameterBlockConstant(&exp_coefficients[0]);
  ceres::Solve(options, &new_problem_two, &summary);
  std::cout << summary.BriefReport() << std::endl;
  set_shape_coef(shape_coefficients);
  set_texture_coef(tex_coefficients);
  write_model("face.off");

  // fourth optmization step

  ceres::Problem full_problem;
  const int full_num_coeff = 80;
  const int full_num_tex_coeff = 80;
  shape_coefficients.resize(full_num_coeff);
  tex_coefficients.resize(full_num_tex_coeff);
  for (unsigned int i = 0; i < num_vertices; ++i) {
    ceres::CostFunction *cost_function_full =
        new ceres::AutoDiffCostFunction<Constraint::PhotoConstraint, 3, 6, 2,
                                        full_num_coeff, full_num_coeff, full_num_tex_coeff, 27,
                                        4>(new Constraint::PhotoConstraint(
            shape_mean, shape_pc, tex_mean, tex_pc, exp_mean, exp_pc,
            shape_variance, tex_variance, exp_variance, image, i, full_num_coeff,
            grid, interpolator, vertices_faces_map, faces));
    full_problem.AddResidualBlock(
        cost_function_full, nullptr, &cam_extrinsics[0], &cam_intrinsics[0],
        &shape_coefficients[0], &exp_coefficients[0], &tex_coefficients[0],
        &light_coefficients[0], &distortion_coefficients[0]);
  }

  // set other constraints
  ceres::CostFunction *shape_prior_full =
      new ceres::AutoDiffCostFunction<Constraint::RegularizationConstraint,
                                      full_num_coeff, full_num_coeff>(
          new Constraint::RegularizationConstraint(full_num_coeff, shape_variance, 1));

  ceres::CostFunction *tex_prior_full =
      new ceres::AutoDiffCostFunction<Constraint::RegularizationConstraint,
                                      full_num_tex_coeff, full_num_tex_coeff>(
          new Constraint::RegularizationConstraint(full_num_tex_coeff, tex_variance, 1));

  ceres::CostFunction *exp_prior_full =
      new ceres::AutoDiffCostFunction<Constraint::RegularizationConstraint,
                                      full_num_coeff, full_num_coeff>(
          new Constraint::RegularizationConstraint(full_num_coeff, exp_variance, 1));

  ceres::CostFunction *dist_prior_full =
      new ceres::AutoDiffCostFunction<Constraint::DistortionRegularizationConstraint,
                                      4, 4>(new Constraint::DistortionRegularizationConstraint(1.0));

  

  
  full_problem.AddResidualBlock(tex_prior_full, nullptr, &tex_coefficients[0]);
  full_problem.AddResidualBlock(shape_prior_full, nullptr, &shape_coefficients[0]);
  full_problem.AddResidualBlock(exp_prior_full, nullptr, &exp_coefficients[0]);
  full_problem.AddResidualBlock(dist_prior_full, nullptr, &distortion_coefficients[0]);
  

  full_problem.SetParameterBlockConstant(&exp_coefficients[0]);
  ceres::Solve(options, &full_problem, &summary);
  std::cout << summary.BriefReport() << std::endl;

  set_shape_coef(shape_coefficients);
  set_texture_coef(tex_coefficients);
  // set_expression_coef(exp_coefficients);
  write_model("face.off");

  // fifth optmization step

  ceres::Problem last_problem;
  for (unsigned int i = 0; i < num_vertices; ++i) {
    ceres::CostFunction *cost_function_full =
        new ceres::AutoDiffCostFunction<Constraint::PhotoConstraint, 3, 6, 2,
                                        full_num_coeff, full_num_coeff, full_num_tex_coeff, 27,
                                        4>(new Constraint::PhotoConstraint(
            shape_mean, shape_pc, tex_mean, tex_pc, exp_mean, exp_pc,
            shape_variance, tex_variance, exp_variance, image, i, full_num_coeff,
            grid, interpolator, vertices_faces_map, faces));
    last_problem.AddResidualBlock(
        cost_function_full, nullptr, &cam_extrinsics[0], &cam_intrinsics[0],
        &shape_coefficients[0], &exp_coefficients[0], &tex_coefficients[0],
        &light_coefficients[0], &distortion_coefficients[0]);
  }

  // set other constraints
  ceres::CostFunction *shape_prior_last =
      new ceres::AutoDiffCostFunction<Constraint::RegularizationConstraint,
                                      full_num_coeff, full_num_coeff>(
          new Constraint::RegularizationConstraint(full_num_coeff, shape_variance, 1.0));

  ceres::CostFunction *tex_prior_last =
      new ceres::AutoDiffCostFunction<Constraint::RegularizationConstraint,
                                      full_num_tex_coeff, full_num_tex_coeff>(
          new Constraint::RegularizationConstraint(full_num_tex_coeff, tex_variance, 1.0));

  ceres::CostFunction *exp_prior_last =
      new ceres::AutoDiffCostFunction<Constraint::RegularizationConstraint,
                                      full_num_coeff, full_num_coeff>(
          new Constraint::RegularizationConstraint(full_num_coeff, exp_variance, 1.0));

  

  
  last_problem.AddResidualBlock(tex_prior_last, nullptr, &tex_coefficients[0]);
  last_problem.AddResidualBlock(shape_prior_last, nullptr, &shape_coefficients[0]);
  last_problem.AddResidualBlock(exp_prior_last, nullptr, &exp_coefficients[0]);
  

  last_problem.SetParameterBlockConstant(&exp_coefficients[0]);
  last_problem.SetParameterBlockConstant(&cam_extrinsics[0]);
  last_problem.SetParameterBlockConstant(&cam_intrinsics[0]);
  last_problem.SetParameterBlockConstant(&light_coefficients[0]);
  last_problem.SetParameterBlockConstant(&distortion_coefficients[0]);
  ceres::Solve(options, &last_problem, &summary);
  std::cout << summary.BriefReport() << std::endl;

  set_shape_coef(shape_coefficients);
  set_texture_coef(tex_coefficients);
  // set_expression_coef(exp_coefficients);
  write_model("face.off");

  std::ofstream outFile("estimated_parameters.txt");
  for (int i = 0; i < 6; ++i) {
    outFile << cam_extrinsics[i] << std::endl;
  }
  outFile << cam_intrinsics[0] << std::endl;
  outFile << distortion_coefficients[0] << std::endl;
  outFile << distortion_coefficients[1] << std::endl;
  outFile << distortion_coefficients[2] << std::endl;
  outFile << distortion_coefficients[3] << std::endl;
  outFile.close();
  std::cout << "Completed" << std::endl;
}
