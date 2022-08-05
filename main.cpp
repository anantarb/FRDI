#include "include/Eigen.h"
#include "include/landmarks.h"
#include "include/model.h"
#include "include/obj_loader.h"
#include "include/preprocessor.h"
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

// Test Eigen
void test_eigen() {
  MatrixXd x(2, 2);
  x(0, 0) = 3;
  x(1, 0) = 2.5;
  x(0, 1) = -1;
  x(1, 1) = x(1, 0) + x(0, 1);
  cout << x << endl;
}

// Test BSM
void test_BSM() {
  // Testing BSM
  std::string BfmH5Path = "../../Data/model2019_bfm.h5";
  std::string LandmarkPath = "../../Data/landmarks_indices.txt";
  BFMModel *m = new BFMModel(BfmH5Path, LandmarkPath);

  std::string ImagePath = "../../Data/test.jpg";
  std::string LandmarksTxtPath = "../../Data/landmarks_te.txt";
  Landmarks::detect(ImagePath, LandmarksTxtPath);

  m->fit_image(LandmarksTxtPath, ImagePath);

  // Testing face and landmarks
  m->write_model("face.off");
  // m->write_landmarks("landmarks.off");
}

// Test Image
void test_image() {
  std::string filepath = "../../Data/test_5.jpg";
  Image image = Image(filepath);
  Image fisheye = image.to_fisheye("fisheye.jpg");
  fisheye.show();
  fisheye.save();
}

// Test Video
void test_video() {
  // std::string filepath = "../../Data/Florence
  // Dataset/subject_01/Video/PTZ-Indoor.mjpeg";
  std::string filepath =
      "../../Data/Florence Dataset/subject_01/Video/Indoor-Cooperative.mjpg";
  Video video = Video(filepath);
  // Video fisheye = video.to_fisheye("test.avi");
  video.show();
}

// Test 3D Object Loader
void test_obj_loader() {
  objl::Loader obj;
  obj.LoadFile("../../Data/Florence "
               "Dataset/subject_01/Model/frontal1/obj/110920150452.obj");
}

// Test FlorenceDataloader
void test_florence_dataloader() {
  FlorenceDataloader dataset =
      FlorenceDataloader("../../Data/Florence Dataset/");
  FlorenceData data = dataset.load_subject(1);

  Video video = Video(data.video_filepath);
  video.show();
}

int main() {
  // test_eigen();
  test_BSM();
  // test_image();
  // test_video();
  // test_obj_loader();
  // test_florence_dataloader();
  return 0;
}