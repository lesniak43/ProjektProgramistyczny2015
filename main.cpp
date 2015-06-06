#include "utils.h"

using namespace std;
using namespace cv;

const int MAX_ALLOWED_ENERGY = 437000000;

int main()
{
    Mat frame = Mat::zeros(480, 640, CV_64F);
    Mat face_image = Mat::zeros(480, 640, CV_64F);

    CascadeClassifier face_cascade;

    int reset_counter = 0;
    double alpha = 1;

    double tx = 300;
    double ty = 250;
    double scale = 430;
    double angle = 0;
    double energy = 27000000000;

    double t_tx = 380;
    double t_ty = 200;
    double t_scale = 660;
    double t_angle = 0;
    double t_energy = 0;
    
    VideoCapture cap;
    cap.open(-1);
    if(!cap.isOpened())
    {
        std::cout << "Błąd kamery.";
        return 1;
    }
    if(!face_cascade.load("data/haarcascade_frontalface_alt.xml"))
    {
        cout<<"Nie znaleziono bazy danych detektora twarzy!"<<endl;
        return 1;
    }

    Mat default_shape = load_2D_mat("data/reference_shape.txt");
    default_shape = default_shape.reshape(0, default_shape.cols / 2);
    vector<Mat> shapes = load_shapes_vector("data/shapes.txt");
    PCA pca = load_fs_pca("data/pca.fs");
    vector<Point> face_points = load_2D_points_vector("data/points_inside_shape.txt");
    Mat R;
    FileStorage fs("data/R.fs", FileStorage::READ);
    fs["R"] >> R;
    fs.release();

    Mat pca_parameters = Mat::zeros(1, pca.eigenvalues.rows, CV_64F);
    Mat t_pca_parameters = Mat::zeros(1, pca.eigenvalues.rows, CV_64F);
    
    Mat triangles = load_triangles();
    Mat model3D = load_model3D();

    //START MAIN LOOP
    while(true)
    {
        energy = 270000000000;
        for(int i=0; i<6; i++) cap>>face_image;

        while(true)
        {
            Mat show_face = face_image.clone();
            Mat reconstructed_data = pca.backProject(pca_parameters);
            Mat reconstructed_shape = data_vector_to_shape(reconstructed_data, default_shape.rows*default_shape.cols);
            Mat reconstructed_texture = data_vector_to_texture(reconstructed_data, default_shape.rows*default_shape.cols);
            reconstructed_shape = move_shape(reconstructed_shape, tx, ty, scale, angle);
            draw_shape(show_face, reconstructed_shape);
            
            //POSIT        
            std::vector<CvPoint3D32f> modelPoints;
            for(int i=0; i<58; i++)
            {
                modelPoints.push_back(cvPoint3D32f(model3D.at<double>(i,0), model3D.at<double>(i,1), model3D.at<double>(i,2)));
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

            //CHECK IF NEW ESTIMATES ARE BETTER
            Mat texture_vector = create_texture_vector(face_image, reconstructed_shape, default_shape, face_points, triangles);
            Mat m_energy = ((texture_vector - reconstructed_texture) * (texture_vector - reconstructed_texture).t());
            t_energy = m_energy.at<double>(0,0);
            if(t_energy < energy)
            {
                t_tx = tx;
                t_ty = ty;
                t_scale = scale;
                t_angle = angle;
                for(int c = 0; c < pca_parameters.cols; c++) t_pca_parameters.at<double>(0,c) = pca_parameters.at<double>(0,c);
                energy = t_energy;
                imshow("Face Detector", show_face);
                imshow("POSIT", show_head);
            }
            else
            {
                tx = t_tx;
                ty = t_ty;
                scale = t_scale;
                angle = t_angle;
                for(int c = 0; c < pca_parameters.cols; c++)
                {
                    pca_parameters.at<double>(0,c) = t_pca_parameters.at<double>(0,c);
                }
                if(alpha == 1) alpha = 1.5;
                else if(alpha == 1.5) alpha = 0.5;
                else if(alpha == 0.5) alpha = 0.25;
                if(alpha == 0.25) alpha = 0.125;
                else
                {
                    alpha = 1;
                    if(energy < MAX_ALLOWED_ENERGY) break;
                    else
                    {
                        Mat find_face = face_image.clone();
                        cvtColor(find_face, find_face, CV_BGR2GRAY);
                        equalizeHist(find_face, find_face);
                        vector<Rect> faces;
                        face_cascade.detectMultiScale(find_face, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30));
                        if(faces.size() > 0)
                        {
                            tx = faces[0].x + faces[0].width*0.5;
                            ty = faces[0].y + faces[0].height*0.5;
                            scale = faces[0].width*2;
                        }
                        else
                        {
                            tx = 300;
                            ty = 250;
                            scale = 430;
                        }
                        angle = 0;
                        t_tx = tx;
                        t_ty = ty;
                        t_angle = angle;
                        t_scale = scale;
                        for(int c = 0; c < t_pca_parameters.cols; c++)
                        {
                            t_pca_parameters.at<double>(0,c) = 0;
                            pca_parameters.at<double>(0,c) = 0;
                        }
                        cout<<"Error too high ("<<energy<<"), face position reset."<<endl;
                        break;
                    }
                }
            }

            //CALCULATE NEW SHAPE, ANGLE AND POSITION
            Mat outcome = R * (texture_vector - reconstructed_texture).t();
            outcome = outcome * alpha;
            int c=0;
            for(; c<pca.eigenvalues.rows; c++) pca_parameters.at<double>(0,c) -= outcome.at<double>(0,c);
            tx -= outcome.at<double>(0,c);
            ty -= outcome.at<double>(0,c+1);
            scale -= outcome.at<double>(0,c+2);
            angle -= outcome.at<double>(0,c+3);

            //PRESS ESC TO QUIT
            if ((char)waitKey(50) == 27) return 0;
        }
    }
}


