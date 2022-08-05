#include "../include/preprocessor.h"

#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Static Methods
cv::Mat To_Fisheye(cv::Mat img, float r_scale, float xc, float yc,
                   bool is_cropped) {
  cv::Mat fisheye_img = img.clone();
  unsigned short width = img.cols;
  unsigned short height = img.rows;
  fisheye_img = cv::Mat::zeros(height, width, img.type());

  int min_x = width;
  int min_y = height;
  int max_x = 0;
  int max_y = 0;

  for (unsigned int x = 0; x < width; x++) {
    for (unsigned int y = 0; y < height; y++) {
      // Normalization
      double norm_x = (double(x) / double(width)) * 2 - 1;
      double norm_y = (double(y) / double(height)) * 2 - 1;

      // Dash
      double dash_x = norm_x * sqrt(1.0f - (pow(norm_y, 2.0f) / 2.0f));
      double dash_y = norm_y * sqrt(1.0f - (pow(norm_x, 2.0f) / 2.0f));

      // Radial Distortion
      double r = r_scale * sqrt(pow(dash_x - xc, 2) + pow(dash_y - yc, 2));

      // Dash Dash
      double dash_dash_x = dash_x * exp(-pow(r, 2.0f) / 4.0f);
      double dash_dash_y = dash_y * exp(-pow(r, 2.0f) / 4.0f);

      // Un-norm
      int un_norm_x = ((dash_dash_x + 1) / 2.0f) * width;
      int un_norm_y = ((dash_dash_y + 1) / 2.0f) * height;

      // Save to new Image
      if (x > width / 2)
        if (fisheye_img.at<cv::Vec3b>(un_norm_y, un_norm_x) ==
            cv::Vec3b(0, 0, 0))
          fisheye_img.at<cv::Vec3b>(un_norm_y, un_norm_x) =
              img.at<cv::Vec3b>(y, x);
      if (x < width / 2)
        fisheye_img.at<cv::Vec3b>(un_norm_y, un_norm_x) =
            img.at<cv::Vec3b>(y, x);

      // Crop Coords
      if (un_norm_x < min_x)
        min_x = un_norm_x;
      if (un_norm_y < min_y)
        min_y = un_norm_y;
      if (un_norm_x > max_x)
        max_x = un_norm_x;
      if (un_norm_y > max_y)
        max_y = un_norm_y;
    }
  }

  if (is_cropped)
    return fisheye_img(cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y));
  else
    return fisheye_img;
}

// Image Class
// Constructors
Image::Image(std::string _filepath) : filepath(_filepath) {
  img = cv::imread(filepath, cv::IMREAD_COLOR);

  if (img.empty()) {
    std::cout << "Image not Open: " << filepath << std::endl;
    return;
  }

  width = img.cols;
  height = img.rows;
}
Image::Image(std::string _filepath, cv::Mat _img)
    : filepath(_filepath), img(_img), width(img.cols), height(img.rows) {}
// Deconstructors
Image::~Image() { img.release(); }
// Methods
void Image::save() { save(filepath); }
void Image::save(std::string _filepath) { cv::imwrite(_filepath, img); }
void Image::show() {
  cv::imshow("Image", img);

  char key = (char)cv::waitKey(0);
  // if (c == 27)
  //	break;
}
Image Image::to_fisheye(std::string _filepath) {
  Image fisheye_img = Image(_filepath, To_Fisheye(img, 1.4, 0, 0, true));
  return fisheye_img;
}
// Methods; Getters
const std::string Image::get_filepath() const { return filepath; }
const cv::Mat Image::get_img() const { return img; }
const unsigned short Image::get_width() const { return width; }
const unsigned short Image::get_height() const { return height; }
// Image Class END

// Video Class
// Constructors
Video::Video(std::string _filepath) : filepath(_filepath) {
  cap.open(filepath);

  if (!cap.isOpened()) {
    std::cout << "Video not Open: " << filepath << std::endl;
    return;
  }

  cv::Mat frame;
  cap >> frame;
  width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
}
// Deconstructors
Video::~Video() { cap.release(); }
// Methods
void Video::save() { save(filepath); }
void Video::save(std::string _filepath) {
  cv::VideoWriter wri(_filepath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                      10, cv::Size(width, height));

  while (true) {
    cv::Mat frame;
    cap >> frame;

    if (frame.empty())
      break;

    wri << frame;

    // 27 = ESC Key
    char key = (char)cv::waitKey(25);
    if (key == 27)
      break;
  }

  wri.release();
}
void Video::show() {
  while (true) {
    cv::Mat frame;
    cap >> frame;

    if (frame.empty())
      break;

    cv::imshow("Video", frame);

    // 27 = ESC Key
    char key = (char)cv::waitKey(25);
    if (key == 27)
      break;
  }

  cv::destroyAllWindows();
}
Video Video::to_fisheye(std::string _filepath) {
  cv::VideoWriter wri(_filepath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                      10, cv::Size(width, height));

  while (true) {
    cv::Mat frame;
    cap >> frame;

    if (frame.empty())
      break;

    wri << To_Fisheye(frame);
  }

  wri.release();

  return Video(_filepath);
}
// Methods; Getters
const std::string Video::get_filepath() const { return filepath; }
const cv::VideoCapture Video::get_video() const { return cap; }
const unsigned short Video::get_width() const { return width; }
const unsigned short Video::get_height() const { return height; }
// Video Class END

// EurecomDataloader Class
// EurecomDataloader Class END

// FlorenceDataloader Class
FlorenceDataloader::FlorenceDataloader(std::string _root_dir)
    : root_dir(_root_dir) {
  std::ifstream file;
  std::string files_file = root_dir + "/files.txt";

  file.open(files_file);
  std::string line;

  if (!file.is_open()) {
    std::cout << "File not Open: " << files_file << std::endl;
    return;
  }

  unsigned int count = 0;
  while (std::getline(file, line))
    files.push_back(root_dir + line);
}
// Methods
FlorenceData FlorenceDataloader::load_subject(int subject_number,
                                              const std::string view_type,
                                              const std::string video_type) {
  std::string subject = "subject_";
  if (subject_number < 10)
    subject += "0" + std::to_string(subject_number);
  else
    subject += std::to_string(subject_number);

  FlorenceData data;

  for (unsigned int i = 0; i < files.size(); i++)
    if (files[i].find(subject) != std::string::npos) {
      if (files[i].find(".obj") != std::string::npos)
        data.obj_filepath = files[i];

      if (files[i].find(video_type) != std::string::npos)
        data.video_filepath = files[i];
    }

  return data;
}
// Methods; Getters
std::string FlorenceDataloader::get_root_dir() { return root_dir; }
std::vector<std::string> FlorenceDataloader::get_files() { return files; }
// FlorenceDataloader Class END