#include "utils.h"

using namespace std;
using namespace cv;

int main()
{
    Mat triangles = load_triangles();

// FIND LIST OF POINTS INSIDE REFERENCE SHAPE
    vector<Point>hull;
    convexHull(mat_shape_to_vector_of_points(load_2D_mat("data/reference_shape.txt")), hull);
    Mat points_inside = Mat::zeros(480, 640, CV_32SC1);
    Point points[hull.size()];
    for(int i=0; i<hull.size(); i++) points[i] = hull[i];
    fillConvexPoly(points_inside, points, hull.size(), Scalar(255));
    vector<Point> points_inside_shape;
    for(int r=0; r<480; r++)
        for(int c=0; c<640; c++)
        {   
            if(points_inside.at<int>(r,c)==255) points_inside_shape.push_back(Point(r,c));
        }
    save_2D_points_vector("data/points_inside_shape.txt", points_inside_shape);

// CREATE VECTOR 'SHAPES' CONSISTING OF (points_per_shape x 2) MATRICES
    Mat default_shape = load_2D_mat("data/reference_shape.txt");
    default_shape = default_shape.reshape(0, default_shape.cols / 2);
    vector<Mat> shapes = load_shapes_vector("data/shapes.txt");

// STANDARIZE ALL SHAPES
    Mat Procrustes_mean;
    Mat result = standarize_shapes(shapes, Procrustes_mean);
    
// NORMALIZE AND SAVE TEXTURES
    Mat all_textures = normalized_textures_matrix(shapes, triangles, default_shape);
    save_2D_mat("data/textures.txt", all_textures);

// CALCULATE PCA
    Mat all_data = Mat::zeros(all_textures.rows, all_textures.cols + result.cols, result.type());
    for(int r=0; r<result.rows; r++)
    {
        for(int c=0; c<result.cols; c++) all_data.at<double>(r,c) = result.at<double>(r,c);
        for(int c=0; c<all_textures.cols; c++) all_data.at<double>(r,result.cols+c) = all_textures.at<double>(r,c);
    }
    PCA pca(all_data, Mat(), CV_PCA_DATA_AS_ROW, 0.95);
    save_fs_pca("data/pca.fs", pca);

// CALCULATE LINEAR MODEL
    cout<<"starting..."<<endl;
    Mat delta_texture, delta_parameters;
    vector<Point> face_points = load_2D_points_vector("data/points_inside_shape.txt");
    delta_texture = Mat::zeros(0, 3*face_points.size(), CV_64F);
    for(int i=0; i<shapes.size(); i++)
    {
        cout<<"starting image "<<i+1<<endl;
        stringstream ss;
        ss<<"images/"<<i+1<<".jpg";
        Mat face_image = imread(ss.str(), CV_LOAD_IMAGE_COLOR);
        calculate_delta_texture(pca, Procrustes_mean, face_image, shapes[i], default_shape, face_points, triangles, delta_texture, delta_parameters);
    }
    cout<<"done!"<<endl;
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
