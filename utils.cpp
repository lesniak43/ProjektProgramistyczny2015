#include <string.h>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>

#include "utils.h"

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <cvaux.h>
#include <cmath>

using namespace std;
using namespace cv;

//LOAD AND SAVE DATA

Mat load_2D_mat(const char* fileName)
{
    FILE* in = fopen(fileName,"r");
    int a;
    int rows, cols;
    fscanf(in,"%d%d",&rows,&cols);
    Mat result = Mat::zeros(rows, cols, CV_64F);
    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<cols; j++)
        {
            fscanf(in,"%d",&a);
            result.at<double>(i,j) = a;
        }
    }
    fclose(in);
    return result;
}

void save_2D_mat(const char* fileName, const Mat& matrix)
{
    ofstream matrix_file;
    matrix_file.open(fileName, ios::trunc);
    matrix_file << matrix.rows << " " << matrix.cols << "\n";
    for(int i=0; i<matrix.rows; i++){
        for(int j=0; j<matrix.cols; j++){
            matrix_file << matrix.at<double>(i,j) << " ";
        }
        matrix_file << "\n";
    }
    matrix_file.close();
}

Mat load_triangles()
{
    FILE* in = fopen("data/triangles.txt", "r");
    int a;
    int rows, cols;
    fscanf(in,"%d%d",&rows,&cols);
    Mat result = Mat::zeros(rows, cols, CV_32S);
    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<cols; j++)
        {
            fscanf(in,"%d",&a);
            result.at<int>(i,j) = a;
        }
    }
    fclose(in);
    return result;
}

Mat load_model3D()
{
    FILE* in = fopen("data/model3D.txt", "r");
    double a;
    int rows, cols;
    fscanf(in,"%d%d",&rows,&cols);
    Mat result = Mat::zeros(rows, cols, CV_64F);
    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<cols; j++)
        {
            fscanf(in,"%lf",&a);
            result.at<double>(i,j) = a;
        }
    }
    fclose(in);
    return result;
}

vector<Mat> load_shapes_vector(const char* fileName)
{
    vector<Mat> shapes;
    FILE* in = fopen(fileName,"r");
    int a;
    int rows, cols;
    fscanf(in,"%d%d",&rows,&cols);
    
    for(int i=0;i<rows;i++){
        Mat shape = Mat::eye(cols/2, 2, CV_64F);
        shapes.push_back(shape);
        for(int j=0;j<cols/2;j++){
            fscanf(in,"%d",&a);
            shapes.at(i).at<double>(j,0) = a;
            fscanf(in,"%d",&a);
            shapes.at(i).at<double>(j,1) = a;
        }
    }
    fclose(in);
    return shapes;
}

vector<Point> load_2D_points_vector(const char* fileName)
{
    FILE* in = fopen(fileName,"r");
    int x, y, size;
    fscanf(in, "%d", &size);
    vector<Point> result;
    for(int i=0; i<size; i++)
    {
        fscanf(in,"%d%d",&x, &y);
        result.push_back(Point(x, y));
    }
    fclose(in);
    return result;
}

void save_2D_points_vector(const char* fileName, const vector<Point>& v_points)
{
    ofstream points_file;
    points_file.open(fileName, ios::trunc);
    points_file << v_points.size() << "\n";
    for(int i=0; i<v_points.size(); i++) points_file << v_points[i].x << " " << v_points[i].y << "\n";
    points_file.close();
}

void save_face_points(const char* fileName, const Mat& default_shape)
{
    vector<Point> face_hull;
    vector<Point> v_default_shape;
    for(int i=0; i<default_shape.rows; i++) v_default_shape.push_back(Point(default_shape.at<double>(i,0),default_shape.at<double>(i,1)));
    convexHull(v_default_shape, face_hull);
    Mat binary_face = Mat::zeros(480, 640, CV_64F);
    fillConvexPoly(binary_face, &face_hull[0], face_hull.size(), 255);
    vector<Point> face_points;
    for(int x=0; x<binary_face.cols; x++) for(int y=0; y<binary_face.rows; y++)
    {
        if(binary_face.at<double>(x,y)==255) face_points.push_back(Point(x,y));
    }
    save_2D_points_vector(fileName, face_points);
}

Mat load_texture_vector(const string fileName, const Mat& shape, const Mat& default_shape, const Mat triangles)
{
    vector<Point> face_points = load_2D_points_vector("data/points_inside_shape.txt");
    Mat face_image = imread(fileName, CV_LOAD_IMAGE_COLOR);
    return create_texture_vector(face_image, shape, default_shape, face_points, triangles);
}

PCA load_fs_pca(const string & fileName)
{
    PCA pca;
    FileStorage fs(fileName, FileStorage::READ);
    fs["mean"] >> pca.mean;
    fs["e_vectors"] >> pca.eigenvectors;
    fs["e_values"] >> pca.eigenvalues;
    fs.release();
    return pca;
}

void save_fs_pca(const string & fileName, cv::PCA pca)
{
    FileStorage fs(fileName, FileStorage::WRITE);
    fs << "mean" << pca.mean;
    fs << "e_vectors" << pca.eigenvectors;
    fs << "e_values" << pca.eigenvalues;
    fs.release();
}

//CONVERT DATA

Mat shape_and_texture_to_data_vector(const Mat& tex_vector, const Mat& shape)
{
    //texture vector - 1 x n
    //shape - m x 2
    //data vector - 1 x 2m+n

    Mat data_vector = Mat::zeros(1, 2*shape.rows+tex_vector.cols, CV_64F);
    int r = 0;
    for(; r<shape.rows; r++)
    {
        data_vector.at<double>(0,2*r) = shape.at<double>(r,0);
        data_vector.at<double>(0,2*r+1) = shape.at<double>(r,1);
    }
    for(int c=0; c<tex_vector.cols; c++) data_vector.at<double>(0,2*r+c) = tex_vector.at<double>(0,c); 
    return data_vector;
}

Mat data_vector_to_texture(const Mat& data_vector, int shape_size)
{
    Mat result = Mat::zeros(1, data_vector.cols - shape_size, CV_64F);
    for(int c=shape_size; c<data_vector.cols; c++) result.at<double>(0,c-shape_size) = data_vector.at<double>(0,c); 
    return result;
}

Mat data_vector_to_shape(const Mat& data_vector, int shape_size)
{   
    Mat reconstructed_shape = Mat::zeros(shape_size/2, 2, CV_64F);

    for(int r=0; r<shape_size/2; r++) for(int c=0; c<2; c++)
        {
            reconstructed_shape.at<double>(r,c) = data_vector.at<double>(0,2*r+c);
        }
    
    return reconstructed_shape;
}

vector<Point> mat_shape_to_vector_of_points(const Mat& shape)
{
    vector<Point> result;
    for(int p=0; p < shape.cols/2; p++)
    {
        result.push_back(Point(shape.at<double>(0,2*p), shape.at<double>(0,2*p+1)));
    }
    return result;
}




int find_nearest_point(std::vector< cv::Point2f > corners, Point2f point, float max_dist)
{
    int index = corners.size();
    float smallest_dist = max_dist;
    float current_dist = max_dist;
    for(int i = 0; i < corners.size(); i++)
    {
        current_dist = abs(corners[i].x - point.x) + abs(corners[i].y - point.y);
        if(current_dist < smallest_dist)
        {
            index = i;
            smallest_dist = current_dist;
        }
    }
    return index;
}

void borderTwoTriangles(int maxwidth, int maxheight, Point2f srcTri[3], Point2f dstTri[3], Rect & srcRect, Rect & dstRect, Point2f n_srcTri[3], Point2f n_dstTri[3])
{
    int srcl=maxwidth-1;
    int srcr=0;
    int srcu=maxheight-1;
    int srcd=0;
    int dstl=maxwidth-1;
    int dstr=0;
    int dstu=maxheight-1;
    int dstd=0;
    
    for(int i=0; i<3; i++)
    {
        if(srcl > srcTri[i].x) srcl = srcTri[i].x;
        if(srcr < srcTri[i].x) srcr = srcTri[i].x;
        if(srcu > srcTri[i].y) srcu = srcTri[i].y;
        if(srcd < srcTri[i].y) srcd = srcTri[i].y;
    }
    for(int i=0; i<3; i++)
    {
        if(dstl > dstTri[i].x) dstl = dstTri[i].x;
        if(dstr < dstTri[i].x) dstr = dstTri[i].x;
        if(dstu > dstTri[i].y) dstu = dstTri[i].y;
        if(dstd < dstTri[i].y) dstd = dstTri[i].y;
    }
    
    if(srcr-srcl > dstr-dstl)
    {
        if(dstr < srcr-srcl)
        {
            dstl = 0;
            dstr = srcr-srcl;
        }
        else dstl = dstr - srcr + srcl;
    }
    if(srcd-srcu > dstd-dstu)
    {
        if(dstd < srcd-srcu)
        {
            dstu = 0;
            dstd = srcd-srcu;
        }
        else dstu = dstd - srcd + srcu;
    }
    if(dstr-dstl > srcr-srcl)
    {
        if(srcr < dstr-dstl)
        {
            srcl = 0;
            srcr = dstr-dstl;
        }
        else srcl = srcr - dstr + dstl;
    }
    if(dstd-dstu > srcd-srcu)
    {
        if(srcd < dstd-dstu)
        {
            srcu = 0;
            srcd = dstd-dstu;
        }
        else srcu = srcd - dstd + dstu;
    }
    
    for(int i=0; i<3; i++)
    {
        n_srcTri[i].x = srcTri[i].x - srcl;
        n_srcTri[i].y = srcTri[i].y - srcu;
        n_dstTri[i].x = dstTri[i].x - dstl;
        n_dstTri[i].y = dstTri[i].y - dstu;
    }
    srcRect = Rect(Point2f(srcl-1,srcu-1), Point2f(srcr+1,srcd+1));
    dstRect = Rect(Point2f(dstl-1,dstu-1), Point2f(dstr+1,dstd+1));
}

void warpTextureFromTriangle(Point2f srcTri[3], const Mat& originalImage, Point2f dstTri[3], Mat& warp_final)
{
    Mat warp_mat( 2, 3, CV_32FC1 );
    Mat warp_dst, warp_mask;

    Rect srcRect, dstRect;
    Point2f n_srcTri[3];
    Point2f n_dstTri[3];

    CvPoint trianglePoints[3];
    trianglePoints[0] = dstTri[0];
    trianglePoints[1] = dstTri[1];
    trianglePoints[2] = dstTri[2];
    warp_dst  = Mat::zeros( originalImage.rows, originalImage.cols, originalImage.type() );
    warp_mask = Mat::zeros( originalImage.rows, originalImage.cols, originalImage.type() );

    borderTwoTriangles(originalImage.cols, originalImage.rows, srcTri, dstTri, srcRect, dstRect, n_srcTri, n_dstTri);
    Mat originalROI(originalImage, srcRect & Rect(0,0,640,480));
    Mat dstROI(warp_dst, dstRect & Rect(0,0,640,480));
    warp_mat = getAffineTransform( n_srcTri, n_dstTri );
    warpAffine( originalROI, dstROI, warp_mat, dstROI.size() );

    cvFillConvexPoly( new IplImage(warp_mask), trianglePoints, 3, CV_RGB(255,255,255), CV_AA, 0 );    
    warp_dst.copyTo(warp_final,warp_mask);
}



Mat standarize_shapes(vector<Mat> arg_shapes, Mat& Procrustes_mean)
{
    vector<Mat> shapes;
    for(int m=0; m<arg_shapes.size(); m++)
    {
        shapes.push_back(arg_shapes[m].clone());
    }
    Mat shape_mean;

    int rows = shapes.at(0).rows;
    for(int i = 0; i < shapes.size(); i++)
    {
        reduce(shapes.at(i), shape_mean, 0, CV_REDUCE_AVG);
        shapes.at(i) = shapes.at(i) - repeat(shape_mean, rows, 1);
        shapes.at(i) = shapes.at(i) / norm(shapes.at(i));
    }

    Mat result(0, 2*rows, CV_64F);
    shape_mean = shapes.at(0).clone();
    Mat w, u, vt;
    
    for(int i = 0; i < shapes.size(); i++)
    {
        SVDecomp( shape_mean.reshape(1).t() * shapes.at(i).reshape(1) , w, u, vt);
        shapes.at(i) = (shapes.at(i).reshape(1) * vt.t()) * u.t();
    }

    //second pass, should be enough (?)
    shape_mean = Mat::zeros(shapes.at(0).rows, shapes.at(0).cols, CV_64F);
    for(int i = 0; i < shapes.size(); i++) shape_mean = shape_mean + shapes.at(i);
    shape_mean = shape_mean / shapes.size();

    for(int i = 0; i < shapes.size(); i++)
    {
        SVDecomp( shape_mean.reshape(1).t() * shapes.at(i).reshape(1) , w, u, vt);
        Mat final_row = (shapes.at(i).reshape(1) * vt.t()) * u.t();
        final_row = final_row.reshape(0, 1);
        result.push_back(final_row);
 
    }
    Procrustes_mean = shape_mean.clone();
    return result;
}

void get_scale_coordinates(const Mat& Procrustes_mean, const Mat& shape, double& tx, double& ty, double& scale, double& angle)
{
    Mat t_vec;
    reduce(shape, t_vec, 0, CV_REDUCE_AVG);
    tx = t_vec.at<double>(0,0);
    ty = t_vec.at<double>(0,1);
    
    Mat translated = shape - repeat(t_vec, shape.rows, 1);
    scale = norm(translated);
    
    translated = translated / scale;

    Mat w, u, vt;

    SVDecomp( Procrustes_mean.reshape(1).t() * translated.reshape(1) , w, u, vt);
    w = (vt.t() * u.t());
    double bla = w.at<double>(0,0);
    if(bla > 1.0) bla = 1.0;
    angle = acos(bla);
}

Mat move_shape(const Mat& shape, const double& tx, const double& ty, const double& scale, const double& angle)
{
    Mat linear = Mat::zeros(2,2,CV_64F);
    linear.at<double>(0,0) = scale * cos(angle);
    linear.at<double>(0,1) = - scale * sin(angle);
    linear.at<double>(1,0) = scale * sin(angle);
    linear.at<double>(1,1) = scale * cos(angle);

    Mat t_vec = Mat::zeros(1,2,CV_64F);
    t_vec.at<double>(0,0) = tx;
    t_vec.at<double>(0,1) = ty;

    Mat result = shape.clone();
    result = (linear * result.t()).t() + repeat(t_vec, result.rows, 1);
    return result;
}

void move_shape(vector<Point>& shape, const double& tx, const double& ty, const double& scale, const double& angle)
{
    Mat m_shape = Mat::zeros(shape.size(),2,CV_64F);
    for(int r=0; r<shape.size(); r++)
    {
        m_shape.at<double>(r,0) = shape[r].x;
        m_shape.at<double>(r,1) = shape[r].y;
    }
    m_shape = move_shape(m_shape, tx, ty, scale, angle);
    for(int r=0; r<shape.size(); r++)
    {
        shape[r].x = m_shape.at<double>(r,0);
        shape[r].y = m_shape.at<double>(r,1);
    }
}


Mat normalized_textures_matrix(const vector<Mat>& shapes, const Mat triangles, const Mat& default_shape)
{
    vector<Point> face_points = load_2D_points_vector("data/points_inside_shape.txt");
    Mat result = Mat::zeros(0, 3 * face_points.size(), CV_64F);
    for(int i = 0; i < shapes.size(); i++)
    {
        stringstream ss;
        ss<<"images/"<<i+1<<".jpg";
        result.push_back(load_texture_vector(ss.str(), shapes[i], default_shape, triangles));
    }
    return result;
}


Mat create_texture_vector(const Mat& face_image, const Mat& shape, const Mat& default_shape, const vector<Point>& face_points, const Mat triangles)
{
    Mat image = face_image.clone();

    vector<Point> temp_shape;
    for(int j=0; j<shape.rows; j++) temp_shape.push_back(Point(shape.at<double>(j,0), shape.at<double>(j,1)));
    Rect boundRect = boundingRect(temp_shape) & Rect(0,0,640,480);

    Mat ycrcb;
    cvtColor(image,ycrcb,CV_BGR2YCrCb);
    vector<Mat> channels;
    split(ycrcb,channels);
    Mat t2roi = channels[0](boundRect);
//    Mat t2roi = channels[0];
    equalizeHist(t2roi, t2roi);
    merge(channels,ycrcb);
    cvtColor(ycrcb,image,CV_YCrCb2BGR);
//    cvtColor(image, image, CV_BGR2GRAY);
//    cvtColor(image, image, CV_GRAY2BGR);
    

    Mat standarized_face = Mat::zeros(image.rows, image.cols, image.type() );
    for(int t=0; t<triangles.rows; t++)
    {
        Point2f srcTri[3] = {temp_shape.at(triangles.at<int>(t,0)), temp_shape.at(triangles.at<int>(t,1)), temp_shape.at(triangles.at<int>(t,2))};            
        Point2f dstTri[3];
        dstTri[0] = Point2f(default_shape.at<double>(triangles.at<int>(t,0), 0), default_shape.at<double>(triangles.at<int>(t,0), 1));
        dstTri[1] = Point2f(default_shape.at<double>(triangles.at<int>(t,1), 0), default_shape.at<double>(triangles.at<int>(t,1), 1));
        dstTri[2] = Point2f(default_shape.at<double>(triangles.at<int>(t,2), 0), default_shape.at<double>(triangles.at<int>(t,2), 1));
        warpTextureFromTriangle(srcTri, image, dstTri, standarized_face);
    }

    Mat result = Mat::zeros(1, 3 * face_points.size(), CV_64F);
    for(int p=0; p<face_points.size(); p++)
    {
        result.at<double>(0, 3*p) = standarized_face.at<Vec3b>(face_points[p].x, face_points[p].y)[0];
        result.at<double>(0, 3*p + 1) = standarized_face.at<Vec3b>(face_points[p].x, face_points[p].y)[1];
        result.at<double>(0, 3*p + 2) = standarized_face.at<Vec3b>(face_points[p].x, face_points[p].y)[2];
    }
    return result;
}

void deform_shape(Mat& input_image, const vector<Point>& input_shape, Mat& output_image, const vector<Point>& output_shape, const Mat triangles)
{
    for(int t=0; t<triangles.rows; t++)
    {
        Point2f srcTri[3] = {input_shape.at(triangles.at<int>(t,0)), input_shape.at(triangles.at<int>(t,1)), input_shape.at(triangles.at<int>(t,2))};            
        Point2f dstTri[3] = {output_shape.at(triangles.at<int>(t,0)), output_shape.at(triangles.at<int>(t,1)), output_shape.at(triangles.at<int>(t,2))};
        warpTextureFromTriangle(srcTri, input_image, dstTri, output_image);
    }
}

void calculate_delta_texture_helper(const Mat& tex_vector, const Mat& reconstructed_data, const Mat& face_image, const Mat& default_shape, const vector<Point>& face_points, const Mat triangles, double tx, double ty, double scale, double angle, Mat& delta_texture)
{
    Mat reconstructed_shape = Mat::zeros(default_shape.size(), default_shape.type());
    for(int r=0; r<default_shape.rows; r++) for(int c=0; c<default_shape.cols; c++)
    {
        reconstructed_shape.at<double>(r,c) = reconstructed_data.at<double>(0,r*default_shape.cols+c);
    }
    Mat tex_vector2 = data_vector_to_texture(reconstructed_data, 2*default_shape.rows);
    Mat distorted_shape = move_shape(reconstructed_shape, tx, ty, scale, angle);
    Mat distorted_tex_vector = create_texture_vector(face_image, distorted_shape, default_shape, face_points, triangles);
    Mat delta_texture_row = distorted_tex_vector - tex_vector2;
    delta_texture.push_back(delta_texture_row);
}



void calculate_delta_texture(const PCA& pca, const Mat& Procrustes_mean, const Mat& face_image, const Mat& shape, const Mat& default_shape, const vector<Point>& face_points, const Mat triangles, Mat& delta_texture, Mat& delta_parameters)
{
    double tx, ty, scale, angle;
    get_scale_coordinates(Procrustes_mean, shape, tx, ty, scale, angle);
    Mat tex_vector = create_texture_vector(face_image, shape, default_shape, face_points, triangles);
    Mat pca_coordinates = pca.project(shape_and_texture_to_data_vector(tex_vector, shape));

    Mat delta_parameters_row;
    Mat reconstructed_data = pca.backProject(pca_coordinates);
    Mat reconstructed_tex_vector = data_vector_to_texture(reconstructed_data, 2*shape.rows);

    //SCALE
    double dscale = 0.1*scale;
    
    for(int j=1; j<=1; j++) for(int h_sign=0; h_sign<2; h_sign++)
    {
        int sign = 2*h_sign-1;

        calculate_delta_texture_helper(reconstructed_tex_vector, reconstructed_data, face_image, default_shape, face_points, triangles, tx, ty, scale + j*sign*dscale, angle, delta_texture);
        delta_parameters_row = Mat::zeros(1, pca_coordinates.cols + 4, CV_64F);
        delta_parameters_row.at<double>(0, pca_coordinates.cols + 2) = j*sign*dscale;
        delta_parameters.push_back(delta_parameters_row);
        cout<<delta_texture.size()<<" delta scale: "<<j*sign*dscale<<endl;
    }

    //ROTATION
    double drotate = 0.08;
    
    for(int j=1; j<=2; j++) for(int h_sign=0; h_sign<2; h_sign++)
    {
        int sign = 2*h_sign-1;

        calculate_delta_texture_helper(reconstructed_tex_vector, reconstructed_data, face_image, default_shape, face_points, triangles, tx, ty, scale, angle+j*sign*drotate, delta_texture);
        delta_parameters_row = Mat::zeros(1, pca_coordinates.cols + 4, CV_64F);
        delta_parameters_row.at<double>(0, pca_coordinates.cols + 3) = j*sign*drotate;
        delta_parameters.push_back(delta_parameters_row);
        cout<<delta_texture.size()<<" delta rotate: "<< j*sign*drotate<<endl;
    }

    //TRANSLATION
    int dxy = 10;
    
    for(int j=1; j<=2; j++) for(int h_sign=0; h_sign<2; h_sign++)
    {
        int sign = 2*h_sign-1;

        calculate_delta_texture_helper(reconstructed_tex_vector, reconstructed_data, face_image, default_shape, face_points, triangles, tx+sign*j*dxy, ty, scale, angle, delta_texture);
        delta_parameters_row = Mat::zeros(1, pca_coordinates.cols + 4, CV_64F);
        delta_parameters_row.at<double>(0, pca_coordinates.cols) = sign*j*dxy;
        delta_parameters.push_back(delta_parameters_row);
        cout<<delta_texture.size()<<" delta x: "<<sign*j*dxy<<endl;

        calculate_delta_texture_helper(reconstructed_tex_vector, reconstructed_data, face_image, default_shape, face_points, triangles, tx, ty+sign*j*dxy, scale, angle, delta_texture);
        delta_parameters_row = Mat::zeros(1, pca_coordinates.cols + 4, CV_64F);
        delta_parameters_row.at<double>(0, pca_coordinates.cols + 1) = sign*j*dxy;
        delta_parameters.push_back(delta_parameters_row);
        cout<<delta_texture.size()<<" delta y: "<<sign*j*dxy<<endl;
    }

    //DEFORMATION
    double dsigma = 0.2;

    for(int i=0; i<pca_coordinates.cols; i++) for(int j=1; j<=2; j++) for(int h_sign=0; h_sign<2; h_sign++)
    {
        int sign = 2*h_sign-1;
        
        Mat distorted_pca_coordinates = pca_coordinates.clone();
        distorted_pca_coordinates.at<double>(0,i) += sign * j * dsigma * sqrt(pca.eigenvalues.at<double>(i,0));
        delta_parameters_row = Mat::zeros(1, pca_coordinates.cols + 4, CV_64F);
        delta_parameters_row.at<double>(0, i) += sign * j * dsigma * sqrt(pca.eigenvalues.at<double>(i,0));
        delta_parameters.push_back(delta_parameters_row);
        reconstructed_data = pca.backProject(distorted_pca_coordinates);
        calculate_delta_texture_helper(tex_vector, reconstructed_data, face_image, default_shape, face_points, triangles, tx, ty, scale, angle, delta_texture);
        cout<<delta_texture.size()<<" delta parameter "<<i<<": "<<sign * j * dsigma * sqrt(pca.eigenvalues.at<double>(i,0))<<endl;
    }
}


void draw_shape(Mat& image, const Mat& shape)
{
    for(unsigned int r = 0; r < shape.rows; r++)
    {
        circle(image, Point(shape.at<double>(r,0), shape.at<double>(r,1)), 1, Scalar(255,255,255), 3);
    }
}
