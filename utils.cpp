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

Mat load_2D_mat_f(const char* fileName)
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
            fscanf(in,"%f",&a);
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
    srcRect = Rect(Point2f(srcl,srcu), Point2f(srcr,srcd));
    dstRect = Rect(Point2f(dstl,dstu), Point2f(dstr,dstd));
}

void warpTextureFromTriangle(Point2f srcTri[3], const Mat& originalImage, Point2f dstTri[3], Mat& warp_final){

/*  //  const clock_t begin_time = clock();
    Mat warp_mat( 2, 3, CV_32FC1 );
    Mat warp_dst, warp_mask;
    CvPoint trianglePoints[3];
    trianglePoints[0] = dstTri[0];
    trianglePoints[1] = dstTri[1];
    trianglePoints[2] = dstTri[2];
    warp_dst  = Mat::zeros( originalImage.rows, originalImage.cols, originalImage.type() );
    warp_mask = Mat::zeros( originalImage.rows, originalImage.cols, originalImage.type() );
    warp_mat = getAffineTransform( srcTri, dstTri );
  //  cout<<float(clock()-begin_time)/CLOCKS_PER_SEC<<" ";
    warpAffine( originalImage, warp_dst, warp_mat, warp_dst.size() );
  //  cout<<float(clock()-begin_time)/CLOCKS_PER_SEC<<" ";
    cvFillConvexPoly( new IplImage(warp_mask), trianglePoints, 3, CV_RGB(255,255,255), CV_AA, 0 );    
  //  cout<<float(clock()-begin_time)/CLOCKS_PER_SEC<<" ";
    warp_dst.copyTo(warp_final,warp_mask);
 //   cout<<float(clock()-begin_time)/CLOCKS_PER_SEC<<endl;


*/
//    const clock_t begin_time = clock();
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
//    cout<<float(clock()-begin_time)/CLOCKS_PER_SEC<<endl;

    borderTwoTriangles(originalImage.cols, originalImage.rows, srcTri, dstTri, srcRect, dstRect, n_srcTri, n_dstTri);
    Mat originalROI(originalImage, srcRect);
    Mat dstROI(warp_dst, dstRect);
    warp_mat = getAffineTransform( n_srcTri, n_dstTri );
    warpAffine( originalROI, dstROI, warp_mat, dstROI.size() );
//    cout<<float(clock()-begin_time)/CLOCKS_PER_SEC<<endl;
//    warp_mat = getAffineTransform( srcTri, dstTri );
//    warpAffine( originalImage, warp_dst, warp_mat, warp_dst.size() );
//    cout<<float(clock()-begin_time)/CLOCKS_PER_SEC<<endl;

    cvFillConvexPoly( new IplImage(warp_mask), trianglePoints, 3, CV_RGB(255,255,255), CV_AA, 0 );    
//    cout<<float(clock()-begin_time)/CLOCKS_PER_SEC<<endl;
    warp_dst.copyTo(warp_final,warp_mask);
//    warp_dst.copyTo(warp_final);
//    cout<<float(clock()-begin_time)/CLOCKS_PER_SEC<<endl;
//    waitKey(1000);
}

Mat equalizeIntensity(const Mat& inputImage)
{
    // http://stackoverflow.com/questions/15007304/histogram-equalization-not-working-on-color-image-opencv
    
    if(inputImage.channels() >= 3)
    {
        Mat ycrcb;

        cvtColor(inputImage,ycrcb,CV_BGR2YCrCb);

        vector<Mat> channels;
        split(ycrcb,channels);

        equalizeHist(channels[0], channels[0]);

        Mat result;
        merge(channels,ycrcb);

        cvtColor(ycrcb,result,CV_YCrCb2BGR);

        return result;
    }
    return Mat();
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

Mat standarize_shapes(vector<Mat>& shapes, Mat& Procrustes_mean)
{
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

Mat normalized_textures_matrix(const vector<Mat>& shapes, const int triangles[95][3], const Mat& default_shape)
{
    vector<Point> face_points = load_2D_points_vector("data/face_points.txt");
    Mat result = Mat::zeros(0, 3 * face_points.size(), CV_64F);
    for(int i = 0; i < shapes.size(); i++)
    {
        stringstream ss;
        ss<<"images/"<<i+1<<".jpg";
        result.push_back(load_texture_vector(ss.str(), shapes[i], default_shape, triangles));
    }
    return result;
}

Mat load_texture_vector(const string fileName, const Mat& shape, const Mat& default_shape, const int triangles[95][3])
{
    vector<Point> face_points = load_2D_points_vector("data/face_points.txt");
    Mat face_image = imread(fileName, CV_LOAD_IMAGE_COLOR);
    return texture_vector(face_image, shape, default_shape, face_points, triangles);
}

Mat texture_vector(const Mat& face_image, const Mat& shape, const Mat& default_shape, const vector<Point>& face_points, const int triangles[95][3])
{
    Mat image = face_image.clone();

    vector<Point> temp_shape;
    for(int j=0; j<shape.rows; j++) temp_shape.push_back(Point(shape.at<double>(j,0), shape.at<double>(j,1)));
    Rect boundRect = boundingRect(temp_shape) & Rect(0,0,640,480);

    cout<<boundRect<<endl;
    Mat ycrcb;
    cvtColor(image,ycrcb,CV_BGR2YCrCb);
    vector<Mat> channels;
    split(ycrcb,channels);
    Mat t2roi = channels[0](boundRect);
    equalizeHist(t2roi, t2roi);
    merge(channels,ycrcb);
    cvtColor(ycrcb,image,CV_YCrCb2BGR);

    Mat standarized_face = Mat::zeros(image.rows, image.cols, image.type() );
    
    for(int t=0; t<95; t++)
    {
        Point2f srcTri[3] = {temp_shape.at(triangles[t][0]), temp_shape.at(triangles[t][1]), temp_shape.at(triangles[t][2])};            
        Point2f dstTri[3];
        dstTri[0] = Point2f(default_shape.at<double>(triangles[t][0], 0), default_shape.at<double>(triangles[t][0], 1));
        dstTri[1] = Point2f(default_shape.at<double>(triangles[t][1], 0), default_shape.at<double>(triangles[t][1], 1));
        dstTri[2] = Point2f(default_shape.at<double>(triangles[t][2], 0), default_shape.at<double>(triangles[t][2], 1));
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

/*
    Mat binary_face2 = Mat::zeros(480, 640, CV_64F);
    vector<Point> face_points2 = load_2D_points_vector("data/face_points.txt");
    for(int i=0; i<face_points2.size(); i++) binary_face2.at<double>(face_points2[i].x, face_points2[i].y) = 255;
    imshow("bb", binary_face2);
*/

void deform_shape(Mat& input_image, const vector<Point>& input_shape, Mat& output_image, const vector<Point>& output_shape, const int triangles[95][3])
{
//    const clock_t begin_time = clock();
    for(int t=0; t<95; t++)
    {
        Point2f srcTri[3] = {input_shape.at(triangles[t][0]), input_shape.at(triangles[t][1]), input_shape.at(triangles[t][2])};            
        Point2f dstTri[3] = {output_shape.at(triangles[t][0]), output_shape.at(triangles[t][1]), output_shape.at(triangles[t][2])};
        warpTextureFromTriangle(srcTri, input_image, dstTri, output_image);
        //cout<<t<<endl;
    }
//    std::cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC<<endl;
}

void eigenscale(Mat& first_dataset, Mat& scale_this)
{
    Mat covar, mean;
    reduce(first_dataset, mean, 0, CV_REDUCE_AVG);
    double sc1 = norm(first_dataset, repeat(mean, first_dataset.rows, 1));

    reduce(scale_this, mean, 0, CV_REDUCE_AVG);
    double sc2 = norm(scale_this, repeat(mean, scale_this.rows, 1));
    
  //  scale_this = repeat(mean, scale_this.rows, 1) + (sc1*sc1/(sc2*sc2)) * (scale_this - repeat(mean, scale_this.rows, 1));
    
    scale_this = repeat(mean, scale_this.rows, 1) + (scale_this - repeat(mean, scale_this.rows, 1));
    
    reduce(first_dataset, mean, 0, CV_REDUCE_AVG);
  //  first_dataset = (first_dataset - repeat(mean, first_dataset.rows, 1));
//    scale_this = (scale_this - repeat(mean, scale_this.rows, 1));

}

void calculate_delta_texture_helper(const Mat& tex_vector, const Mat& reconstructed_data, const Mat& face_image, const Mat& default_shape, const vector<Point>& face_points, int triangles[95][3], double tx, double ty, double scale, double angle, Mat& delta_texture)
{
    Mat reconstructed_shape = Mat::zeros(default_shape.size(), default_shape.type());
    for(int r=0; r<default_shape.rows; r++) for(int c=0; c<default_shape.cols; c++)
    {
        reconstructed_shape.at<double>(r,c) = reconstructed_data.at<double>(0,r*default_shape.cols+c);
    }
    Mat distorted_shape = move_shape(reconstructed_shape, tx, ty, scale, angle);
    Mat distorted_tex_vector = texture_vector(face_image, distorted_shape, default_shape, face_points, triangles);
    Mat delta_texture_row = distorted_tex_vector - tex_vector;
    delta_texture.push_back(delta_texture_row);
}

void calculate_delta_texture(const PCA& pca, const Mat& Procrustes_mean, const Mat& face_image, const Mat& shape, const Mat& default_shape, const vector<Point>& face_points, int triangles[95][3], Mat& delta_texture, Mat& delta_parameters)
{
    Mat tex_vector = texture_vector(face_image, shape, default_shape, face_points, triangles);
    
    Mat data_vector = Mat::zeros(1, 2*shape.rows+tex_vector.cols, CV_64F);
    int r = 0;
    for(; r<shape.rows; r++)
    {
        data_vector.at<double>(0,2*r) = shape.at<double>(r,0);
        data_vector.at<double>(0,2*r+1) = shape.at<double>(r,1);
    }
    for(int c=0; c<tex_vector.cols; c++) data_vector.at<double>(0,2*r+c) = tex_vector.at<double>(0,c); 
    
    double tx, ty, scale, angle;
    get_scale_coordinates(Procrustes_mean, shape, tx, ty, scale, angle);
/*
c i ± 0.25σ i , ± 0.5σ i
Scale 90%, 110%
θ ±5 o , ±10 o
t x , t y ± 5%, ± 10%
*/
    Mat pca_coordinates = pca.project(data_vector);
    Mat delta_parameters_row;
    
    Mat reconstructed_data = pca.backProject(pca_coordinates);

    calculate_delta_texture_helper(tex_vector, reconstructed_data, face_image, default_shape, face_points, triangles, tx, ty, scale*0.9, angle, delta_texture);
    delta_parameters_row = Mat::zeros(1, pca_coordinates.cols + 4, CV_64F);
    delta_parameters_row.at<double>(0, pca_coordinates.cols + 2) = (-0.1)*scale;
    delta_parameters.push_back(delta_parameters_row);
    cout<<delta_texture.size()<<endl;

    calculate_delta_texture_helper(tex_vector, reconstructed_data, face_image, default_shape, face_points, triangles, tx, ty, scale*1.1, angle, delta_texture);
    delta_parameters_row = Mat::zeros(1, pca_coordinates.cols + 4, CV_64F);
    delta_parameters_row.at<double>(0, pca_coordinates.cols + 2) = (0.1)*scale;
    delta_parameters.push_back(delta_parameters_row);
    cout<<delta_texture.size()<<endl;

    calculate_delta_texture_helper(tex_vector, reconstructed_data, face_image, default_shape, face_points, triangles, tx, ty, scale, angle+0.1, delta_texture);
    delta_parameters_row = Mat::zeros(1, pca_coordinates.cols + 4, CV_64F);
    delta_parameters_row.at<double>(0, pca_coordinates.cols + 3) = 0.1;
    delta_parameters.push_back(delta_parameters_row);
    cout<<delta_texture.size()<<endl;

    calculate_delta_texture_helper(tex_vector, reconstructed_data, face_image, default_shape, face_points, triangles, tx, ty, scale, angle-0.1, delta_texture);
    delta_parameters_row = Mat::zeros(1, pca_coordinates.cols + 4, CV_64F);
    delta_parameters_row.at<double>(0, pca_coordinates.cols + 3) = -0.1;
    delta_parameters.push_back(delta_parameters_row);
    cout<<delta_texture.size()<<endl;

    calculate_delta_texture_helper(tex_vector, reconstructed_data, face_image, default_shape, face_points, triangles, tx, ty, scale, angle+0.2, delta_texture);
    delta_parameters_row = Mat::zeros(1, pca_coordinates.cols + 4, CV_64F);
    delta_parameters_row.at<double>(0, pca_coordinates.cols + 3) = 0.2;
    delta_parameters.push_back(delta_parameters_row);
    cout<<delta_texture.size()<<endl;

    calculate_delta_texture_helper(tex_vector, reconstructed_data, face_image, default_shape, face_points, triangles, tx, ty, scale, angle-0.2, delta_texture);
    delta_parameters_row = Mat::zeros(1, pca_coordinates.cols + 4, CV_64F);
    delta_parameters_row.at<double>(0, pca_coordinates.cols + 3) = -0.2;
    delta_parameters.push_back(delta_parameters_row);
    cout<<delta_texture.size()<<endl;

    calculate_delta_texture_helper(tex_vector, reconstructed_data, face_image, default_shape, face_points, triangles, tx+25, ty, scale, angle, delta_texture);
    delta_parameters_row = Mat::zeros(1, pca_coordinates.cols + 4, CV_64F);
    delta_parameters_row.at<double>(0, pca_coordinates.cols) = 25;
    delta_parameters.push_back(delta_parameters_row);
    cout<<delta_texture.size()<<endl;

    calculate_delta_texture_helper(tex_vector, reconstructed_data, face_image, default_shape, face_points, triangles, tx-25, ty, scale, angle, delta_texture);
    delta_parameters_row = Mat::zeros(1, pca_coordinates.cols + 4, CV_64F);
    delta_parameters_row.at<double>(0, pca_coordinates.cols) = -25;
    delta_parameters.push_back(delta_parameters_row);
    cout<<delta_texture.size()<<endl;

    calculate_delta_texture_helper(tex_vector, reconstructed_data, face_image, default_shape, face_points, triangles, tx+50, ty, scale, angle, delta_texture);
    delta_parameters_row = Mat::zeros(1, pca_coordinates.cols + 4, CV_64F);
    delta_parameters_row.at<double>(0, pca_coordinates.cols) = 50;
    delta_parameters.push_back(delta_parameters_row);
    cout<<delta_texture.size()<<endl;

    calculate_delta_texture_helper(tex_vector, reconstructed_data, face_image, default_shape, face_points, triangles, tx-50, ty, scale, angle, delta_texture);
    delta_parameters_row = Mat::zeros(1, pca_coordinates.cols + 4, CV_64F);
    delta_parameters_row.at<double>(0, pca_coordinates.cols) = -50;
    delta_parameters.push_back(delta_parameters_row);
    cout<<delta_texture.size()<<endl;

    calculate_delta_texture_helper(tex_vector, reconstructed_data, face_image, default_shape, face_points, triangles, tx, ty+25, scale, angle, delta_texture);
    delta_parameters_row = Mat::zeros(1, pca_coordinates.cols + 4, CV_64F);
    delta_parameters_row.at<double>(0, pca_coordinates.cols + 1) = 25;
    delta_parameters.push_back(delta_parameters_row);
    cout<<delta_texture.size()<<endl;

    calculate_delta_texture_helper(tex_vector, reconstructed_data, face_image, default_shape, face_points, triangles, tx, ty-25, scale, angle, delta_texture);
    delta_parameters_row = Mat::zeros(1, pca_coordinates.cols + 4, CV_64F);
    delta_parameters_row.at<double>(0, pca_coordinates.cols + 1) = -25;
    delta_parameters.push_back(delta_parameters_row);
    cout<<delta_texture.size()<<endl;

    calculate_delta_texture_helper(tex_vector, reconstructed_data, face_image, default_shape, face_points, triangles, tx, ty+50, scale, angle, delta_texture);
    delta_parameters_row = Mat::zeros(1, pca_coordinates.cols + 4, CV_64F);
    delta_parameters_row.at<double>(0, pca_coordinates.cols + 1) = 50;
    delta_parameters.push_back(delta_parameters_row);
    cout<<delta_texture.size()<<endl;

    calculate_delta_texture_helper(tex_vector, reconstructed_data, face_image, default_shape, face_points, triangles, tx, ty-50, scale, angle, delta_texture);
    delta_parameters_row = Mat::zeros(1, pca_coordinates.cols + 4, CV_64F);
    delta_parameters_row.at<double>(0, pca_coordinates.cols + 1) = -50;
    delta_parameters.push_back(delta_parameters_row);
    cout<<delta_texture.size()<<endl;

    for(int i=0; i<pca_coordinates.cols; i++) for(int j=1; j<3; j++) for(int sign=0; sign<2; sign++)
    {
        Mat distorted_pca_coordinates = pca_coordinates.clone();
        distorted_pca_coordinates.at<double>(0,i) += (2*sign-1) * j * 0.25 * sqrt(pca.eigenvalues.at<double>(i,0));
        delta_parameters_row = Mat::zeros(1, pca_coordinates.cols + 4, CV_64F);
        delta_parameters_row.at<double>(0, i) += (2*sign-1) * j * 0.25 * sqrt(pca.eigenvalues.at<double>(i,0));
        delta_parameters.push_back(delta_parameters_row);
        reconstructed_data = pca.backProject(distorted_pca_coordinates);
        calculate_delta_texture_helper(tex_vector, reconstructed_data, face_image, default_shape, face_points, triangles, tx, ty, scale, angle, delta_texture);
        cout<<delta_texture.size()<<endl;
    }
}

PCA load_pca(const char * eigenvalues, const char * eigenvectors, const char * mean)
{
    Mat pca_eigenvalues = load_2D_mat_f(eigenvalues);
    Mat pca_eigenvectors = load_2D_mat_f(eigenvectors);
    Mat pca_mean = load_2D_mat_f(mean);

    PCA result;
    result.eigenvalues = pca_eigenvalues;
    result.eigenvectors = pca_eigenvectors;
    result.mean = pca_mean;

    return result;
}

void save_pca(const char * eigenvalues, const char * eigenvectors, const char * mean, const PCA& pca)
{
    save_2D_mat(eigenvalues, pca.eigenvalues);
    save_2D_mat(eigenvectors, pca.eigenvectors);
    save_2D_mat(mean, pca.mean);

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
