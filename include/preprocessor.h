#include <string>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <Eigen/Dense>

#define FRONTAL1 "frontal1"
#define FRONTAL2 "frontal2"
#define SIDER "sider"
#define SIDEL "sidel"

#define COOPERATIVE "Indoor-Cooperative"
#define OUTDOOR "PTZ-Outdoor"
#define INDOOR "PTZ-Indoor"

// Static Method
cv::Mat To_Fisheye(cv::Mat img, float r_scale = 1.0f, float xc = 0.0f, float yc = 0.0f, bool is_cropped = false);

// Image Class
class Image
{
private:
	std::string filepath;
	cv::Mat img;
	unsigned short width;
	unsigned short height;

public:
	// Constructors
	Image(std::string _filepath);					// Load Image
	Image(std::string _filepath, cv::Mat _img);		// Create Image using Existing Mat

	// Deconstructors
	~Image();

	// Methods
	void save();
	void save(std::string new_filepath);
	void show();
	Image to_fisheye(std::string _filepath);

	// Getters
	const std::string get_filepath() const;
	const cv::Mat get_img() const;
	const unsigned short get_width() const;
	const unsigned short get_height() const;
};

// Video Class
class Video
{
private:
	std::string filepath;
	cv::VideoCapture cap;
	unsigned short width;
	unsigned short height;

public:
	// Constructors
	Video(std::string _filepath);

	// Deconstructors
	~Video();

	// Methods
	void save();
	void save(std::string new_filepath);
	void show();
	Video to_fisheye(std::string _filepath);

	// Getters
	const std::string get_filepath() const;
	const cv::VideoCapture get_video() const;
	const unsigned short get_width() const;
	const unsigned short get_height() const;
};

// EurecomDataloader Class
struct FlorenceData
{
	std::string obj_filepath;
	std::string video_filepath;
};

// FlorenceDataloader Class
class FlorenceDataloader
{
private:
	std::string root_dir;
	std::vector<std::string> files;

public:
	// Constructors
	FlorenceDataloader(std::string _root_dir);

	// Methods
	FlorenceData load_subject(
		int subject_number,
		const std::string view_type = FRONTAL1,
		const std::string video_type = COOPERATIVE
	);

	// Getters
	std::string get_root_dir();
	std::vector<std::string> get_files();
};