#include <string.h>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <cvaux.h>

using namespace std;
using namespace cv;

void        mouse_callback( int event, int x, int y, int, void*);
Mat         load_2D_mat(const char* fileName);
Mat         load_2D_mat_f(const char* fileName);
void        save_2D_mat(const char* fileName, const Mat& matrix);
void        borderTwoTriangles(int maxwidth, int maxheight, Point2f srcTri[3], Point2f dstTri[3], Rect & srcRect, Rect & dstRect, Point2f n_srcTri[3], Point2f n_dstTri[3]);
void        warpTextureFromTriangle(Point2f srcTri[3], const Mat& originalImage, Point2f dstTri[3], Mat& warp_final);
Mat         equalizeIntensity(const Mat& inputImage);
vector<Mat> load_shapes_vector(const char* fileName);
int         find_nearest_point(std::vector< cv::Point2f > corners, Point2f point, float max_dist=10);
Mat         standarize_shapes(vector<Mat>& shapes, Mat& Procrustes_mean);
vector<Point>   load_2D_points_vector(const char* fileName);
void        save_2D_points_vector(const char* fileName, const vector<Point>& v_points);
void        save_face_points(const char* fileName, const Mat& default_shape);
Mat         normalized_textures_matrix(const vector<Mat>& shapes, const int triangles[95][3], const Mat& default_shape);
Mat         load_texture_vector(const string fileName, const Mat& shape, const Mat& default_shape, const int triangles[95][3]);
Mat         texture_vector(const Mat& face_image, const Mat& shape, const Mat& default_shape, const vector<Point>& face_points, const int triangles[95][3]);
void        deform_shape(Mat& input_image, const vector<Point>& input_shape, Mat& output_image, const vector<Point>& output_shape, const int triangles[95][3]);
void        eigenscale(Mat& first_dataset, Mat& scale_this);
void        get_scale_coordinates(const Mat& Procrustes_mean, const Mat& shape, double& tx, double& ty, double& scale, double& angle);
Mat         move_shape(const Mat& shape, const double& tx, const double& ty, const double& scale, const double& angle);
void        move_shape(vector<Point>& shape, const double& tx, const double& ty, const double& scale, const double& angle);
void         calculate_delta_texture(const PCA& pca, const Mat& Procrustes_mean, const Mat& face_image, const Mat& shape, const Mat& default_shape, const vector<Point>& face_points, int triangles[95][3], Mat& delta_texture, Mat& delta_parameters);
PCA         load_pca(const char * eigenvalues, const char * eigenvectors, const char * mean);
void        save_pca(const char * eigenvalues, const char * eigenvectors, const char * mean, const PCA& pca);
PCA         load_fs_pca(const string & fileName);
void        save_fs_pca(const string & fileName, cv::PCA pca);
