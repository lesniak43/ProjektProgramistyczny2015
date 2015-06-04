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

Mat frame = Mat::zeros(480, 640, CV_64F);

int main()
{
    VideoCapture cap;
    cap.open(-1);
    if(!cap.isOpened())
    {
        std::cout << "Błąd kamery.";
        return 1;
    }
    
    //LOAD MODEL DATA
        // CREATE VECTOR 'SHAPES' CONSISTING OF (points_per_shape x 2) MATRICES
    Mat default_shape = load_2D_mat("data/reference_shape.txt");
    default_shape = default_shape.reshape(0, default_shape.cols / 2);
    vector<Mat> shapes = load_shapes_vector("data/shapes.txt");
    PCA pca = load_fs_pca("data/pca.fs");
    vector<Point> face_points = load_2D_points_vector("data/face_points.txt");

    Mat R;
    FileStorage fs("data/R.fs", FileStorage::READ);
    fs["R"] >> R;
    fs.release();





    bool flag_grab = true;
    bool flag_reset = true;
    int reset_counter = 0;



    //INITIAL TEST DATA
    Mat vvv = Mat::zeros(1, pca.eigenvalues.rows, CV_64F);
    Mat t_vvv = Mat::zeros(1, pca.eigenvalues.rows, CV_64F);

//    double tx = 400;
//    double ty = 250;
//    double scale = 700;
//    double angle = 0;
//    Mat face_image = imread("images/3.jpg", CV_LOAD_IMAGE_COLOR);

    int i_tx = 380;
    int i_ty = 200;
    int i_scale = 660;
    int i_angle = 0;

    Mat face_image = imread("images/1.jpg", CV_LOAD_IMAGE_COLOR);

//    double tx = 340;
//    double ty = 280;
//    double scale = 660;
//    double angle = 0;
//    Mat face_image = imread("images/12.jpg", CV_LOAD_IMAGE_COLOR);

//    double tx = 300;
//    double ty = 230;
//    double scale = 660;
//    double angle = 0;
//    Mat face_image = imread("images/20.jpg", CV_LOAD_IMAGE_COLOR);


    double tx = 380;
    double ty = 200;
    double scale = 660;
    double angle = 0;

    double t_tx = 380;
    double t_ty = 200;
    double t_scale = 660;
    double t_angle = 0;

    
    double alpha = 1;
    double energy = 27000000000;
    double t_energy = 0;
    namedWindow("circ", WINDOW_AUTOSIZE);    
    
    int parameters[pca.eigenvalues.rows];
    int * trackbar_index = parameters;

    createTrackbar( "tx" , "circ", &i_tx, 800);
    createTrackbar( "ty" , "circ", &i_ty, 600);
    createTrackbar( "scale" , "circ", &i_scale, 2000);
    createTrackbar( "angle" , "circ", &i_angle, 360);
    
    for(int i=0; i<pca.eigenvalues.rows; i++)
    {
        parameters[i]=50;
        char name[1] = {(char)(i+(int)'0')};
        createTrackbar( name , "circ", trackbar_index, 100);
        trackbar_index++;
    }
    
    while(true)
    {
        energy = 27000000000;
        
        int face_index = 0;

        if (!flag_grab) while(true)
        {
            imshow("prev", face_image);
            int pressed_key = waitKey(50);

            if ((char)pressed_key == '<') {
                face_index += 43;
                face_index--;
                face_index %= 43;
                stringstream ss;
                ss<<"images/"<<face_index+1<<".jpg";
                face_image = imread(ss.str(), CV_LOAD_IMAGE_COLOR);
            }
            if ((char)pressed_key == '>') {
                face_index++;
                face_index %= 43;
                stringstream ss;
                ss<<"images/"<<face_index+1<<".jpg";
                face_image = imread(ss.str(), CV_LOAD_IMAGE_COLOR);
            }
            if ((char)pressed_key == 27) {
                break;
            }
            if ((char)pressed_key == 'x') {
                return 0;
            }
        }

        if (!flag_grab) while(true)
        {
            Mat show_face = face_image.clone();

            tx = (double)i_tx;
            ty = (double)i_ty;
            scale = (double)i_scale;
            angle = (double)i_angle;
            
            for(int i=0; i<pca.eigenvalues.rows; i++)
            {
                vvv.at<double>(0,i) = ((parameters[i]-50))*10*sqrt(pca.eigenvalues.at<double>(i,0));
            }
            vvv = vvv / 100;


            Mat reconstructed_data = pca.backProject(vvv);
            Mat reconstructed_shape = Mat::zeros(default_shape.size(), default_shape.type());
            Mat reconstructed_texture = Mat::zeros(1, 3*face_points.size(), default_shape.type());
            for(int c=0; c<3*face_points.size(); c++) reconstructed_texture.at<double>(0,c) = reconstructed_data.at<double>(0,default_shape.rows*default_shape.cols+c);

            for(int r=0; r<default_shape.rows; r++) for(int c=0; c<default_shape.cols; c++)
            {
                reconstructed_shape.at<double>(r,c) = reconstructed_data.at<double>(0,r*default_shape.cols+c);
            }        
            reconstructed_shape = move_shape(reconstructed_shape, tx, ty, scale, angle);
            
            vector<Point> face_shape;
            for(int i=0; i<reconstructed_shape.rows; i++) face_shape.push_back(Point(reconstructed_shape.at<double>(i,0),reconstructed_shape.at<double>(i,1)));
            for(unsigned int r = 0; r < face_shape.size(); r++) circle(show_face, face_shape[r], 1, Scalar(255,255,255), 3);
            
            imshow("aaa", show_face);
            
            int pressed_key = waitKey(50);
            if ((char)pressed_key == 27) {
                break;
            }
        }
        
        
        
        
        
        
        if(flag_grab) for(int i=0; i<6; i++) cap>>face_image;

        //START MAIN LOOP
        while(true)
        {
            Mat show_face = face_image.clone();

            Mat reconstructed_data = pca.backProject(vvv);
            Mat reconstructed_shape = Mat::zeros(default_shape.size(), default_shape.type());
            Mat reconstructed_texture = Mat::zeros(1, 3*face_points.size(), default_shape.type());
            for(int c=0; c<3*face_points.size(); c++) reconstructed_texture.at<double>(0,c) = reconstructed_data.at<double>(0,default_shape.rows*default_shape.cols+c);

            for(int r=0; r<default_shape.rows; r++) for(int c=0; c<default_shape.cols; c++)
            {
                reconstructed_shape.at<double>(r,c) = reconstructed_data.at<double>(0,r*default_shape.cols+c);
            }        
            reconstructed_shape = move_shape(reconstructed_shape, tx, ty, scale, angle);
            
            vector<Point> face_shape;
            for(int i=0; i<reconstructed_shape.rows; i++) face_shape.push_back(Point(reconstructed_shape.at<double>(i,0),reconstructed_shape.at<double>(i,1)));
            for(unsigned int r = 0; r < face_shape.size(); r++)
            {
//                cout<<face_shape[r].x<<" "<<face_shape[r].y<<endl;
                circle(show_face, face_shape[r], 1, Scalar(255,255,255), 3);
            }
            cout<<energy<<endl;
            
            imshow("aaa", show_face);
            
            
            //POSIT        
            std::vector<CvPoint3D32f> modelPoints;
            for(int i=0; i<58; i++)
            {
                modelPoints.push_back(cvPoint3D32f(Model3D[i][0], Model3D[i][1], Model3D[i][2]));
            }
            CvPOSITObject *positObject = cvCreatePOSITObject( &modelPoints[0], static_cast<int>(modelPoints.size()) );

            vector<CvPoint2D32f> srcImagePoints;
            for(int r=0; r<reconstructed_shape.rows; r++) srcImagePoints.push_back(cvPoint2D32f(reconstructed_shape.at<double>(r,0),reconstructed_shape.at<double>(r,1)));

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
                facepoint.at<double>(0,0) = Model3D[i][0] - Model3D[0][0];
                facepoint.at<double>(1,0) = Model3D[i][1] - Model3D[0][1];
                facepoint.at<double>(2,0) = Model3D[i][2] - Model3D[0][2];
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
            
            imshow("3dhead", show_head);


            //CALCULATE NEW SHAPE, ANGLE AND POSITION
            Mat g_image = texture_vector(face_image, reconstructed_shape, default_shape, face_points, triangles);
            
            Mat m_energy = ((g_image - reconstructed_texture) * (g_image - reconstructed_texture).t());
            t_energy = m_energy.at<double>(0,0);
            if(t_energy < energy)
            {
                t_tx = tx;
                t_ty = ty;
                t_scale = scale;
                t_angle = angle;
                for(int c = 0; c < vvv.cols; c++) t_vvv.at<double>(0,c) = vvv.at<double>(0,c);
                energy = t_energy;
            }
            else
            {
                reset_counter++;
                reset_counter %= 7;
                tx = t_tx;
                ty = t_ty;
                scale = t_scale;
                angle = t_angle;
//                for(int c = 0; c < vvv.cols; c++)
//                {
//                    vvv.at<double>(0,c) = (1-(double)c*0.001) * t_vvv.at<double>(0,c);
//                    vvv.at<double>(0,c) = t_vvv.at<double>(0,c);
//                }
//                if(reset_counter==0 && flag_reset)
//                    for(int c = 0; c < vvv.cols; c++)
//                    {
//                        vvv.at<double>(0,c) = 0;
//                    }
                
//                cout<<"done"<<endl;
                
                if(alpha == 0.125) {alpha = 1; break;}
                if(alpha == 0.25) alpha = 0.125;
                if(alpha == 0.5) alpha = 0.25;
                if(alpha == 1.5) alpha = 0.5;
                if(alpha == 1) alpha = 1.5;
            }
            
            Mat outcome = R * (g_image - reconstructed_texture).t();
            outcome = outcome * alpha * 0.3;
            int c=0;
            for(; c<pca.eigenvalues.rows; c++) vvv.at<double>(0,c) -= outcome.at<double>(0,c);
            tx -= outcome.at<double>(0,c);
            ty -= outcome.at<double>(0,c+1);
            scale -= outcome.at<double>(0,c+2);
            angle -= outcome.at<double>(0,c+3);
            
            int pressed_key = waitKey(50);
            if ((char)pressed_key == 'r') {
                vvv = Mat::zeros(1, pca.eigenvalues.rows, CV_64F);
                tx = 380;
                ty = 200;
                scale = 660;
                angle = 0;
                break;
            }
        }
    }

    return 0;

}


