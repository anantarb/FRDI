#include "../include/landmarks.h"

void Landmarks::detect(const std::string ImagePath, const std::string TxtPath) {
	const std::string cmd = "conda run -n 3dfa python ../detect_landmarks.py --image_path " + ImagePath + " --txt_path " + TxtPath;
	int te = system(cmd.c_str());
	return;
}