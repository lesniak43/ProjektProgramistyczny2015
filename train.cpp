#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <string>
#include <algorithm>
#include <set>
#include "utils.h"
#include "consts.cpp"

using namespace std;
using namespace cv;

int main()
{
// CREATE VECTOR 'SHAPES' CONSISTING OF (points_per_shape x 2) MATRICES
    Mat default_shape = load_2D_mat("data/reference_shape.txt");
    default_shape = default_shape.reshape(0, default_shape.cols / 2);
    vector<Mat> shapes = load_shapes_vector("data/shapes.txt");
    vector<Mat> shapes2 = load_shapes_vector("data/shapes.txt");

// STANDARIZE ALL SHAPES
    Mat Procrustes_mean;
    Mat result = standarize_shapes(shapes, Procrustes_mean);
    
// NORMALIZE AND SAVE TEXTURES

    Mat all_textures = normalized_textures_matrix(shapes2, triangles, default_shape);
    save_2D_mat("data/textures.txt", all_textures);

// CALCULATE PCA
//    eigenscale(result, all_textures);

    Mat all_data = Mat::zeros(all_textures.rows, all_textures.cols + result.cols, result.type());
    for(int r=0; r<result.rows; r++)
    {
        for(int c=0; c<result.cols; c++) all_data.at<double>(r,c) = result.at<double>(r,c);
        for(int c=0; c<all_textures.cols; c++) all_data.at<double>(r,result.cols+c) = all_textures.at<double>(r,c);
    }
    PCA pca(all_data, Mat(), CV_PCA_DATA_AS_ROW, 0.95);
    save_fs_pca("data/pca.fs", pca);



    double tx, ty, scale, angle;
    get_scale_coordinates(Procrustes_mean, shapes2[0], tx, ty, scale, angle);
    cout<<"starting..."<<endl;
    Mat delta_texture, delta_parameters;
    vector<Point> face_points = load_2D_points_vector("data/face_points.txt");
    delta_texture = Mat::zeros(0, 3*face_points.size(), CV_64F);

    for(int i=0; i<shapes2.size(); i++)
    {
        cout<<"starting image "<<i+1<<endl;
        stringstream ss;
        ss<<"images/"<<i+1<<".jpg";
        Mat face_image = imread(ss.str(), CV_LOAD_IMAGE_COLOR);
        calculate_delta_texture(pca, Procrustes_mean, face_image, shapes2[i], default_shape, face_points, triangles, delta_texture, delta_parameters);
    }
    cout<<"done!"<<endl;
//    save_2D_mat("data/delta_texture.txt", delta_texture);
//    save_2D_mat("data/delta_parameters.txt", delta_parameters);

//    vector<Point> face_points = load_2D_points_vector("data/face_points.txt");
//    Mat delta_texture = load_2D_mat("data/delta_texture.txt");
//    Mat delta_parameters = load_2D_mat("data/delta_parameters.txt");
    Mat J, R, t1;
    
    invert(delta_parameters.t(), t1, DECOMP_SVD);
    cout<<t1.size()<<endl<<delta_texture.size()<<endl;
    J = delta_texture.t() * t1;
    invert(J, R, DECOMP_SVD);
    save_2D_mat("data/R.txt", R);

    FileStorage fs("data/R.fs", FileStorage::WRITE);
    fs << "R" << R;
    fs.release();

    return 0;
}
