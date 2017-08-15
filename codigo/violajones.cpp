#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

 #include <iostream>
 #include <stdio.h>

 using namespace std;
 using namespace cv;

 /** Function Headers */
 void detectAndDisplay( Mat frame );

 /** Global variables */
 String face_cascade_name = "/home/thecakeisalai/opencv/opencv-3.0.0-alpha/data/haarcascades/haarcascade_frontalface_alt.xml";
 String eyes_cascade_name = "/home/thecakeisalai/opencv/opencv-3.0.0-alpha/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
 String nose_cascade_name = "/home/thecakeisalai/opencv/opencv-3.0.0-alpha/data/haarcascades/haarcascade_mcs_nose.xml";
 String mouth_cascade_name = "/home/thecakeisalai/opencv/opencv-3.0.0-alpha/data/haarcascades/haarcascade_mcs_mouth.xml";
 String teste_name = "/home/thecakeisalai/opencv/opencv-3.0.0-alpha/data/haarcascades/haarcascade_frontalface_alt.xml";
 CascadeClassifier face_cascade;
 CascadeClassifier eyes_cascade;
 CascadeClassifier nose_cascade;
 CascadeClassifier mouth_cascade;
 CascadeClassifier teste;
 string window_name = "Capture - Face detection";
 RNG rng(12345);
 Point pt[4];
 /** @function main */
/*
 int main( int argc, const char** argv )
 {
   VideoCapture capture = VideoCapture("out.mp4");
   //VideoCapture capture;
   Mat frame;

   //-- 1. Load the cascades
   if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face\n"); return -1; };
   if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading eye\n"); return -1; };
   if( !nose_cascade.load( nose_cascade_name ) ){ printf("--(!)Error loading nose\n"); return -1; };
   if( !mouth_cascade.load( mouth_cascade_name ) ){ printf("--(!)Error loading mouth\n"); return -1; };
   if( !teste.load( teste_name ) ){ printf("--(!)Error loading teste\n"); return -1; };
   //-- 2. Read the video stream
    //capture.open(0);
    namedWindow("MyVideo");
     while(true)
     {
         waitKey(100);
        capture >> frame;
   //     imshow("antes",frame);
   //     frame = imread("/home/thecakeisalai/Desktop/index.jpg");

   //-- 3. Apply the classifier to the frame
       if( !frame.empty() )
        detectAndDisplay( frame );
       else
       { printf(" --(!) No captured frame -- Break!"); break; }

       int c = waitKey(10);
       if( (char)c == 'c' ) { break; }
        if(!frame.empty()){
            cout << "teste 1" << endl;
            imshow("MyVideo", frame);
        }
        else {
            cout << "AQUI";
        }
      }
   return 0;
 }
*/

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
  std::vector<Rect> faces;
  Mat frame_gray;

  cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );

  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0, Size(30, 30) );
  if(face_cascade.empty() == true){
      printf("aqui");
      return;
  }
  for( size_t i = 0; i < faces.size(); i++ )
  {
    Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
    if(frame.rows < faces[i].height*0.5 + center.y || 0 > center.y - 0.5*faces[i].height)
        continue;
    ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

    Mat faceROI = frame_gray( faces[i] );
    std::vector<Rect> eyes;

    //-- In each face, detect eyes
    eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 , Size(30, 30) );

    for( size_t j = 0; j < eyes.size() && j < 2; j++ )
     {
        printf("\n%d\n",eyes.size());
       Point centerEye( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
       int radiusEye = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
       ellipse( frame,centerEye, Size(eyes[j].width*0.4, eyes[j].height*0.25),0,0,360,Scalar( 255, 255, 255 ), 2, 16, 0);
     }

    //-- In each face, detect nose
    std::vector<Rect> nose;
    nose_cascade.detectMultiScale(faceROI,nose,1.1,2,0,Size(30,30));
    int nose_y;
    for( size_t j = 0; j < nose.size(); j++ )
     {
       Point centerNose( faces[i].x + nose[j].x + nose[j].width*0.5, faces[i].y + nose[j].y + nose[j].height*0.4 );

       if(centerNose.y + nose[j].height*0.35> frame.rows || 0 > centerNose.y - 0.35*nose[j].height) continue;

       ellipse( frame,centerNose, Size(nose[j].width*0.35, nose[j].height*0.35),0,0,360,Scalar( 255, 255, 0 ), 2, 16, 0);
       nose_y = nose[j].y;
     }
    return;
    std::vector<Rect> mouth;
    mouth_cascade.detectMultiScale(faceROI,mouth,1.1,2,0,Size(30,30));
    for( size_t j = 0; j <= mouth.size(); j++ )
     {
       if(nose_y > mouth[j].y)
           continue;
       Point centerMouth( faces[i].x + mouth[j].x + mouth[j].width*0.5, faces[i].y + mouth[j].y + mouth[j].height*0.5 );
      // int radiusMouth = cvRound( (mouth[j].width + mouth[j].height)*0.25 );
       //circle( frame, centerMouth, radiusMouth, Scalar( 255, 0, 0 ), 4, 8, 0 );

       if(centerMouth.y + mouth[j].height*0.25> frame.rows || 0 > centerMouth.y - 0.25*mouth[j].height)

       ellipse( frame,centerMouth, Size(mouth[j].width*0.4, mouth[j].height*0.25),0,0,360,Scalar( 255, 0, 0 ), 2, 16, 0);
     }

    // in each face, detect cheek
    /*Point centerCheekR( faces[i].x + faces[i].width*0.75, faces[i].y + nose[i].y + nose[i].height*0.5  );
    ellipse( frame,centerCheekR, Size((faces[i].width - nose[i].width)*0.25*0.7, (faces[i].width - nose[i].width)*0.25*0.7),0,0,360,Scalar( 0, 0, 0 ), 2, 16, 0);
    Point centerCheekL( faces[i].x + faces[i].width*0.25, faces[i].y + nose[i].y + nose[i].height*0.5  );
    ellipse( frame,centerCheekL, Size((faces[i].width - nose[i].width)*0.25*0.7, (faces[i].width - nose[i].width)*0.25*0.7),0,0,360,Scalar( 0, 0, 0 ), 2, 16, 0);
*/
    int x[1];
    x[0] = 1;
    // in each face, detect forhead
    pt[0] = Point(faces[i].x + faces[i].width*0.5, faces[i].y);// ponto de cima no centro da face
    pt[1] = Point(faces[i].x + faces[i].width*0.5, faces[i].y + eyes[i].y); // ponto de baixo no centro da face
    pt[2] = Point(faces[i].x + eyes[i].x + eyes[i].width*0.5, faces[i].y); // ponto de cima na reta do centro do olho
    pt[3] = Point(faces[i].x + faces[i].width*0.5 + eyes[i].width, faces[i].y + eyes[i].y); // ponto de baixo no centro da face
    //pt[4] = Point(faces[i].x + faces[i].width*0.5, faces[i].y);// ponto de cima no centro da face
    //fillPoly(frame,pt,4,x,Scalar(0,0,255));
    //Point centerFourheadR( faces[i].x + eyes[i].x + eyes[i].width*0.5, faces[i].y);
    //ellipse( frame,centerFourheadR, Size((faces[i].width - nose[i].width)*0.25*0.7, (faces[i].width - nose[i].width)*0.25*0.7),0,0,360,Scalar( 0, 0, 0 ), 2, 16, 0);


  }
  //-- Show what you got
  imshow( window_name, frame );
 }
