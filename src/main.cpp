#include <iostream>
#include <vector>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyData.h>
#include <vtkPLYWriter.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkImageViewer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleImage.h>
#include <vtkLight.h>
#include <vtkLightCollection.h>
#include <vtkRenderer.h>
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkFloatArray.h>
#include <vtkTriangle.h>

using namespace cv;
using namespace std;

void displayMesh(int width, int height, cv::Mat Z) {
    
    /* creating visualization pipeline which basically looks like this:
     vtkPoints -> vtkPolyData -> vtkPolyDataMapper -> vtkActor -> vtkRenderer */
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkPolyDataMapper> modelMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    vtkSmartPointer<vtkActor> modelActor = vtkSmartPointer<vtkActor>::New();
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkCellArray> vtkTriangles = vtkSmartPointer<vtkCellArray>::New();
    
    /* insert x,y,z coords */
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            points->InsertNextPoint(x, y, Z.at<float>(y,x));
        }
    }
    
    /* setup the connectivity between grid points */
    vtkSmartPointer<vtkTriangle> triangle = vtkSmartPointer<vtkTriangle>::New();
    triangle->GetPointIds()->SetNumberOfIds(3);
    for (int i=0; i<height-1; i++) {
        for (int j=0; j<width-1; j++) {
            triangle->GetPointIds()->SetId(0, j+(i*width));
            triangle->GetPointIds()->SetId(1, (i+1)*width+j);
            triangle->GetPointIds()->SetId(2, j+(i*width)+1);
            vtkTriangles->InsertNextCell(triangle);
            triangle->GetPointIds()->SetId(0, (i+1)*width+j);
            triangle->GetPointIds()->SetId(1, (i+1)*width+j+1);
            triangle->GetPointIds()->SetId(2, j+(i*width)+1);
            vtkTriangles->InsertNextCell(triangle);
        }
    }
    polyData->SetPoints(points);
    polyData->SetPolys(vtkTriangles);
    
    /* create two lights */
    vtkSmartPointer<vtkLight> light1 = vtkSmartPointer<vtkLight>::New();
    light1->SetPosition(-1, 1, 1);
    renderer->AddLight(light1);
    vtkSmartPointer<vtkLight> light2 = vtkSmartPointer<vtkLight>::New();
    light2->SetPosition(1, -1, -1);
    renderer->AddLight(light2);
    
    /* meshlab-ish background */
    modelMapper->SetInputData(polyData);
    renderer->SetBackground(.45, .45, .9);
    renderer->SetBackground2(.0, .0, .0);
    renderer->GradientBackgroundOn();
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);
    modelActor->SetMapper(modelMapper);
    
    /* setting some properties to make it look just right */
    modelActor->GetProperty()->SetSpecularColor(1, 1, 1);
    modelActor->GetProperty()->SetAmbient(0.2);
    modelActor->GetProperty()->SetDiffuse(0.2);
    modelActor->GetProperty()->SetInterpolationToPhong();
    modelActor->GetProperty()->SetSpecular(0.8);
    modelActor->GetProperty()->SetSpecularPower(8.0);
    
    renderer->AddActor(modelActor);
    vtkSmartPointer<vtkRenderWindowInteractor> interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    interactor->SetRenderWindow(renderWindow);
    
    /* export mesh */
    vtkSmartPointer<vtkPLYWriter> plyExporter = vtkSmartPointer<vtkPLYWriter>::New();
    plyExporter->SetInputData(polyData);
    plyExporter->SetFileName("export.ply");
    plyExporter->SetColorModeToDefault();
    plyExporter->SetArrayName("Colors");
    plyExporter->Update();
    plyExporter->Write();
    
    /* render mesh */
    renderWindow->Render();
    interactor->Start();
}

cv::Mat globalHeights(cv::Mat Pgrads, cv::Mat Qgrads) {
    
    cv::Mat P(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));
    cv::Mat Q(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));
    cv::Mat Z(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));
    
    float lambda = 1.0f;
    float mu = 1.0f;
    
    cv::dft(Pgrads, P, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(Qgrads, Q, cv::DFT_COMPLEX_OUTPUT);
    for (int i=0; i<Pgrads.rows; i++) {
        for (int j=0; j<Pgrads.cols; j++) {
            if (i != 0 || j != 0) {
                float u = sin((float)(i*2*CV_PI/Pgrads.rows));
                float v = sin((float)(j*2*CV_PI/Pgrads.cols));
                
                float uv = pow(u,2)+pow(v,2);
                float d = (1.0f + lambda)*uv + mu*pow(uv,2);
                Z.at<cv::Vec2f>(i, j)[0] = (u*P.at<cv::Vec2f>(i, j)[1] + v*Q.at<cv::Vec2f>(i, j)[1]) / d;
                Z.at<cv::Vec2f>(i, j)[1] = (-u*P.at<cv::Vec2f>(i, j)[0] - v*Q.at<cv::Vec2f>(i, j)[0]) / d;
            }
        }
    }
    
    /* setting unknown average height to zero */
    Z.at<cv::Vec2f>(0, 0)[0] = 0.0f;
    Z.at<cv::Vec2f>(0, 0)[1] = 0.0f;
    
    cv::dft(Z, Z, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    
    return Z;
}

cv::Vec3f getLightDirFromSphere(Mat Image, Rect boundingbox) {
    
    const int THRESH    = 254;
    const float radius  = boundingbox.width / 2.0f;
    
    Mat Binary;
    threshold(Image, Binary, THRESH, 255, CV_THRESH_BINARY);
    Mat SubImage(Binary, boundingbox);
    
    /* calculate center of pixels */
    Moments m = moments(SubImage, false);
    Point center(m.m10/m.m00, m.m01/m.m00);
    
    /* x,y are swapped here */
    float x = (center.y - radius) / radius;
    float y = (center.x - radius) / radius;
    float z = sqrt(1.0 - pow(x, 2.0) - pow(y, 2.0));
    
    return Vec3f(x, y, z);
}

cv::Rect getBoundingBox(cv::Mat Mask) {
    
    std::vector<std::vector<cv::Point> > v;
    cv::findContours(Mask.clone(), v, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
    assert(v.size() > 0);
    return cv::boundingRect(v[0]);
}

int main(int argc, char *argv[]) {
    
    const int NUM_IMGS          = 12;
    const string CALIBRATION    = "../../images/chrome/chrome.";
    const string MODEL          = "../../images/rock/rock.";
    
    vector<Mat> calibImages;
    vector<Mat> modelImages;
    Mat Lights(NUM_IMGS, 3, CV_32F);
    Mat Mask = imread(CALIBRATION + "mask.png", CV_LOAD_IMAGE_GRAYSCALE);
    Mat ModelMask = imread(MODEL + "mask.png", CV_LOAD_IMAGE_GRAYSCALE);
    Rect bb = getBoundingBox(Mask);
    for (int i = 0; i < NUM_IMGS; i++) {
        Mat Calib = imread(CALIBRATION + to_string(i) + ".png",
                           CV_LOAD_IMAGE_GRAYSCALE);
        Mat tmp = imread(MODEL + to_string(i) + ".png",
                           CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat Model;
        tmp.copyTo(Model, ModelMask);
        Vec3f light = getLightDirFromSphere(Calib, bb);
        Lights.at<float>(i, 0) = light[0];
        Lights.at<float>(i, 1) = light[1];
        Lights.at<float>(i, 2) = light[2];
        calibImages.push_back(Calib);
        modelImages.push_back(Model);
    }
    
    const int height    = calibImages[0].rows;
    const int width     = calibImages[0].cols;
    /* light directions, surface normals, p,q gradients */
    cv::Mat LightsInv;
    cv::invert(Lights, LightsInv, cv::DECOMP_SVD);

    cv::Mat Normals(height, width, CV_32FC3, cv::Scalar::all(0));
    cv::Mat Pgrads(height, width, CV_32F, cv::Scalar::all(0));
    cv::Mat Qgrads(height, width, CV_32F, cv::Scalar::all(0));
    /* estimate surface normals and p,q gradients */
    for (int x=0; x<width; x++) {
        for (int y=0; y<height; y++) {
            Vec<float, NUM_IMGS> I;
            for (int i = 0; i < NUM_IMGS; i++) {
                I[i] = modelImages[i].at<uchar>(Point(x,y));
            }
            
            cv::Mat n = LightsInv * cv::Mat(I);
            float p = sqrt(cv::Mat(n).dot(n));
            if (p > 0) { n = n/p; }
            if (n.at<float>(2,0) == 0) { n.at<float>(2,0) = 1.0; }
            int legit = 1;
            /* avoid spikes ad edges */
            for (int i = 0; i < NUM_IMGS; i++) {
                legit *= modelImages[i].at<uchar>(Point(x,y)) >= 0;
            }
            if (legit) {
                Normals.at<cv::Vec3f>(cv::Point(x,y)) = n;
                Pgrads.at<float>(cv::Point(x,y)) = n.at<float>(0,0)/n.at<float>(2,0);
                Qgrads.at<float>(cv::Point(x,y)) = n.at<float>(1,0)/n.at<float>(2,0);
            } else {
                cv::Vec3f nullvec(0.0f, 0.0f, 1.0f);
                Normals.at<cv::Vec3f>(cv::Point(x,y)) = nullvec;
                Pgrads.at<float>(cv::Point(x,y)) = 0.0f;
                Qgrads.at<float>(cv::Point(x,y)) = 0.0f;
            }
            
        }
    }

    cv::Mat Normalmap;
    cv::cvtColor(Normals, Normalmap, CV_BGR2RGB);
    cv::imshow("Normalmap", Normalmap);
    
    /* global integration of surface normals */
    cv::Mat Z = globalHeights(Pgrads, Qgrads);

    /* display reconstruction */
    displayMesh(Pgrads.cols, Pgrads.rows, Z);

    cv::waitKey();
    return 0;
}