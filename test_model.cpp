#include "utils.h"

using namespace std;
using namespace cv;

int main()
{
    
    Mat triangles = load_triangles();
    Mat model3D = load_model3D();
    
    namedWindow("POSIT", WINDOW_AUTOSIZE);
    namedWindow("Reconstructed face", WINDOW_AUTOSIZE);
    namedWindow("Trackbars", WINDOW_AUTOSIZE);

    PCA pca = load_fs_pca("data/pca.fs");
    int tx = 400;
    int ty = 300;
    int scale = 800;
    double angle = 0;
    int i_angle = 180;
    Mat frame = Mat::zeros(480, 640, CV_64F);
    
    int shape_cols = 116;
    int texture_cols = 221760;
    
    int parameters[pca.eigenvalues.rows];
    int * trackbar_index = parameters;
    for(int i=0; i<pca.eigenvalues.rows; i++)
    {
        parameters[i]=50;
        stringstream ss;
        ss << i;
        createTrackbar( ss.str() , "Trackbars", trackbar_index, 100);
        trackbar_index++;
    }
    createTrackbar( "tx" , "Trackbars", &tx, 800);
    createTrackbar( "ty" , "Trackbars", &ty, 600);
    createTrackbar( "scale" , "Trackbars", &scale, 2000);
    createTrackbar( "angle" , "Trackbars", &i_angle, 360);

    while(true)
    {
        angle = (((double)i_angle)-180)*3.141/180;
        frame = Mat::zeros(480, 640, CV_64F);

        Mat init = Mat::zeros(1, pca.eigenvalues.rows, CV_64F);        
        for(int i=0; i<pca.eigenvalues.rows; i++)
        {
            init.at<double>(0,i) = ((parameters[i]-50))*10*sqrt(pca.eigenvalues.at<double>(i,0));
        }
        init = init / 100;
        Mat reconstructed_data = pca.backProject(init);
        
        Mat reconstructed_shape = Mat::zeros(1, shape_cols, CV_64F);
        Mat reconstructed_texture = Mat::zeros(1, texture_cols, CV_64F);

        for(int c=0; c<shape_cols; c++) reconstructed_shape.at<double>(0,c) = reconstructed_data.at<double>(0,c);
        for(int c=0; c<texture_cols; c++) reconstructed_texture.at<double>(0,c) = reconstructed_data.at<double>(0,shape_cols+c);

        vector<Point> v_reconstructed_shape;
        reconstructed_shape = reconstructed_shape.reshape(0, reconstructed_shape.cols / 2);
        reconstructed_shape = move_shape(reconstructed_shape, tx, ty, scale, angle);
        reconstructed_shape = reconstructed_shape.reshape(0, 1);
        for(int r=0; r<reconstructed_shape.cols/2; r++) v_reconstructed_shape.push_back(Point(reconstructed_shape.at<double>(0,2*r),reconstructed_shape.at<double>(0,2*r+1)));


        Mat rec_face = Mat::zeros(480, 640, CV_8UC3);
        vector<Point> face_points = load_2D_points_vector("data/points_inside_shape.txt");
        Mat m_face_shape = load_2D_mat("data/reference_shape.txt");
        vector<Point> face_shape;
        for(int i=0; i<m_face_shape.cols/2; i++) face_shape.push_back(Point(m_face_shape.at<double>(0,2*i),m_face_shape.at<double>(0,2*i+1)));
        for(int p=0; p<face_points.size(); p++)
        {
            for(int i=0; i<3; i++)
            {
                if(reconstructed_texture.at<double>(0,3*p+i)>255) rec_face.at<Vec3b>(face_points[p].x, face_points[p].y)[i] = 255;
                else if(reconstructed_texture.at<double>(0,3*p+i)<0) rec_face.at<Vec3b>(face_points[p].x, face_points[p].y)[i] = 0;
                else rec_face.at<Vec3b>(face_points[p].x, face_points[p].y)[i] = reconstructed_texture.at<double>(0,3*p+i);
            }
        }
        deform_shape(rec_face, face_shape, frame, v_reconstructed_shape, triangles);
        for(unsigned int r = 0; r < reconstructed_shape.cols/2; r++)
        {
            circle(frame, Point(reconstructed_shape.at<double>(0,2*r), reconstructed_shape.at<double>(0,2*r+1)), 1, Scalar(255), 5);
        }


        //POSIT        
        std::vector<CvPoint3D32f> modelPoints;
        for(int i=0; i<58; i++)
        {
            modelPoints.push_back(cvPoint3D32f(model3D.at<double>(i,0), model3D.at<double>(i,1), model3D.at<double>(i,2)));
        }
        CvPOSITObject *positObject = cvCreatePOSITObject( &modelPoints[0], static_cast<int>(modelPoints.size()) );

        vector<CvPoint2D32f> srcImagePoints;
        for(int r=0; r<reconstructed_shape.cols/2; r++) srcImagePoints.push_back(cvPoint2D32f(reconstructed_shape.at<double>(0,2*r),reconstructed_shape.at<double>(0,2*r+1)));

        CvMatr32f rotation_matrix = new float[9];
        CvVect32f translation_vector = new float[3];
        CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 100, 1.0e-4f);
        cvPOSIT( positObject, &srcImagePoints[0], 10000, criteria, rotation_matrix, translation_vector );
        //focal_length set at 10000 experimentally, so that it works...

        Mat rotate = Mat::zeros(3,3,CV_64F);
        for(int i=0; i<3; i++) for(int j=0; j<3; j++) rotate.at<double>(i,j) = rotation_matrix[3*i+j];

        vector<Point> dots;
        for(int i=0; i<58; i++)
        {
            Mat facepoint = Mat::zeros(3,1,CV_64F);
            facepoint.at<double>(0,0) = model3D.at<double>(i,0) - model3D.at<double>(0,0);
            facepoint.at<double>(1,0) = model3D.at<double>(i,1) - model3D.at<double>(0,1);
            facepoint.at<double>(2,0) = model3D.at<double>(i,2) - model3D.at<double>(0,2);
            facepoint = 15 * rotate * facepoint;
            dots.push_back(Point(facepoint.at<double>(0,0) + translation_vector[0], facepoint.at<double>(1,0) + translation_vector[1]));
        }
        Point mean(0,0);
        for(int i=0; i<dots.size(); i++)
        {
            mean = mean + dots[i];
        }
        mean.x = mean.x / (int)dots.size();
        mean.y = mean.y / (int)dots.size();
        Mat show_head = Mat::zeros(480, 640, CV_8U);
        for(int i=0; i<dots.size(); i++) circle(show_head, dots[i] - mean + Point(320,240), 1, Scalar(255), 3);
        delete rotation_matrix;
        delete translation_vector;

        imshow("POSIT", show_head);
        imshow("Reconstructed face", frame);

        int pressed_key = waitKey(50);
        if ((char)pressed_key == 27) {
            break;
        }
    }
    return 0;
}
