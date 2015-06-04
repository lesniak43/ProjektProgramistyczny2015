#include <string.h>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <fstream>

#include "utils.h"
#include "consts.cpp"

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <cvaux.h>

using namespace std;
using namespace cv;

void mouse_callback( int event, int x, int y, int, void*);

bool drawing_box = false;
bool ready_to_process = true;
bool flag_save = false;
bool flag_stop = true;
bool flag_dragging = false;
bool flag_lines = true;
bool flag_dots = true;
int index_dragged_point = 0;
int snapshot_number = 1;
VideoCapture cap;
Mat frame;
Mat temp;
std::vector< cv::Point2f > corners;


int main(int argc, char** argv)
{
    cap.open(-1);
    if(!cap.isOpened())
    {
        std::cout << "Błąd kamery.";
        return 1;
    }

    Mat shapes = load_2D_mat("data/shapes.txt");
    Mat reference_shape = load_2D_mat("data/reference_shape.txt");

    for(int i = 0; i < shapes.cols/2; i++) corners.push_back(Point2f(shapes.at<double>(snapshot_number-1, 2*i), shapes.at<double>(snapshot_number-1, 2*i+1)));
    flag_stop = true;
    stringstream ss;
    ss<<"images/"<<snapshot_number<<".jpg";
    temp = imread(ss.str(), CV_LOAD_IMAGE_COLOR);
    
    const string window_name = "Create Model";
    namedWindow(window_name, CV_WINDOW_AUTOSIZE);
    setMouseCallback(window_name, mouse_callback);

    while(true)
    {
        if(flag_save)
        {
            flag_save = false;

            stringstream ss;
            ss<<"images/"<<snapshot_number<<".jpg";
            imwrite(ss.str(), temp);

            for(int i = 0; i < shapes.cols/2; i++)
            {
                shapes.at<double>(snapshot_number-1, 2*i) = corners.at(i).x;
                shapes.at<double>(snapshot_number-1, 2*i + 1) = corners.at(i).y;
            }
            save_2D_mat("data/shapes.txt", shapes);
        }
        if(!flag_stop)
        {
            cap.read(frame);
            temp = frame.clone();
        }
        else
        {
            frame = temp.clone();
        }

  //      imshow(window_name, temp);


// DISPLAY FACE TRIANGULATION

        if(flag_dots) for(unsigned int r = 0; r < corners.size(); r++)    circle(frame, corners[r], 1, Scalar(100,255,255), 1);
        if(flag_lines) for(int i = 0; i < 95; i++)
        {
            line(frame, corners[triangles[i][0]], corners[triangles[i][1]], Scalar(255,255,255), 1, 4);
            line(frame, corners[triangles[i][1]], corners[triangles[i][2]], Scalar(255,255,255), 1, 4);
            line(frame, corners[triangles[i][2]], corners[triangles[i][0]], Scalar(255,255,255), 1, 4);
        }
        stringstream ss;
        ss<<snapshot_number<<"/"<<shapes.rows;
        putText(frame, ss.str(), Point(0, 450), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255,255,255), 2);
        imshow(window_name, frame);



// CONTROL SECTION

        int c;
        c = cvWaitKey(30);
        if( (char) c == 27) break;
        if( (char) c == 's' ) flag_save = true;
        if( (char) c == 'c') {flag_stop = !flag_stop;}
        if( (char) c == 'i')
        {
            for(int i = 0; i < corners.size(); i++)
            {
                double tx = corners[i].x - 320;
                double ty = corners[i].y - 240;
                corners[i].x = floor(((0.96) * tx + 0.28 * ty) + 320);
                corners[i].y = floor(((-0.28) * tx + 0.96 * ty) + 240);
            }
        }
        if( (char) c == 'u')
        {
            for(int i = 0; i < corners.size(); i++)
            {
                double tx = corners[i].x - 320;
                double ty = corners[i].y - 240;
                corners[i].x = floor(((0.96) * tx + (-0.28) * ty) + 320);
                corners[i].y = floor((0.28 * tx + 0.96 * ty) + 240);
            }
        }
        if( (char) c == 'j') for(int i = 0; i < corners.size(); i++) corners[i].y = corners[i].y + 5;
        if( (char) c == 'k') for(int i = 0; i < corners.size(); i++) corners[i].y = corners[i].y - 5;
        if( (char) c == 'l') for(int i = 0; i < corners.size(); i++) corners[i].x = corners[i].x + 5;
        if( (char) c == 'h') for(int i = 0; i < corners.size(); i++) corners[i].x = corners[i].x - 5;
        if( (char) c == '>' && snapshot_number < shapes.rows)
        {
            snapshot_number++;
            corners.clear();
            for(int i = 0; i < 58; i++) corners.push_back(Point2f(shapes.at<double>(snapshot_number-1, 2*i), shapes.at<double>(snapshot_number-1, 2*i+1)));
            flag_stop = true;
            stringstream ss;
            ss<<"images/"<<snapshot_number<<".jpg";
            temp = imread(ss.str(), CV_LOAD_IMAGE_COLOR);
        }
        if( (char) c == '<' && snapshot_number > 1)
        {
            snapshot_number--;
            corners.clear();
            for(int i = 0; i < 58; i++) corners.push_back(Point2f(shapes.at<double>(snapshot_number-1, 2*i), shapes.at<double>(snapshot_number-1, 2*i+1)));
            flag_stop = true;
            stringstream ss;
            ss<<"images/"<<snapshot_number<<".jpg";
            temp = imread(ss.str(), CV_LOAD_IMAGE_COLOR);
        }
        if( (char) c == '+')
        {
            Mat new_row = Mat::zeros(1, shapes.cols, CV_64F);
            shapes.push_back(new_row);
            snapshot_number = shapes.rows;
            flag_save = true;
            flag_stop = true;
        }
        if( (char) c == 'e')
        {
            Mat ycrcb;
            cvtColor(temp,ycrcb,CV_BGR2YCrCb);
            vector<Mat> channels;
            split(ycrcb,channels);
            equalizeHist(channels[0], channels[0]);
            merge(channels,ycrcb);
            cvtColor(ycrcb,temp,CV_YCrCb2BGR);
        }
        if( (char) c == 'h') for(int i = 0; i < corners.size(); i++) corners[i].x = corners[i].x - 5;
        if( (char) c == 'z') flag_lines = !flag_lines;
        if( (char) c == 'x') flag_dots = !flag_dots;
    }

// EXIT

    return 0;
}

// MOVE POINTS

void mouse_callback(int event, int x, int y, int, void*)
{
    switch( event )
    {
        case CV_EVENT_MOUSEMOVE: 
            if(flag_dragging)
            {
                corners[index_dragged_point].x = x;
                corners[index_dragged_point].y = y;
            }
            break;

        case CV_EVENT_LBUTTONDOWN:
            index_dragged_point = find_nearest_point(corners, Point2f(x,y));
            if (index_dragged_point < corners.size()) flag_dragging = true;
            break;

        case CV_EVENT_LBUTTONUP:
            flag_dragging = false;
            break;
    }
}
