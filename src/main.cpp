#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyData.h>
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
    modelMapper->SetInput(polyData);
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
    
    /* render mesh */
    renderWindow->Render();
    interactor->Start();
}

cv::Mat globalHeights(cv::Mat Pgrads, cv::Mat Qgrads) {
    
    cv::Mat P(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));
    cv::Mat Q(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));
    cv::Mat Z(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));
    
    cv::dft(Pgrads, P, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(Qgrads, Q, cv::DFT_COMPLEX_OUTPUT);
    for (int i=0; i<Pgrads.rows; i++) {
        for (int j=0; j<Pgrads.cols; j++) {
            if (i != 0 || j != 0) {
                float v = sin(i*2*CV_PI/Pgrads.rows);
                float u = sin(j*2*CV_PI/Pgrads.cols);
                float uv = u*u + v*v;
                float d = (1+0)*uv + 0*uv*uv;
                Z.at<cv::Vec2f>(i, j)[0] = (u*P.at<cv::Vec2f>(i, j)[1] + v*Q.at<cv::Vec2f>(i, j)[1]) / d;
                Z.at<cv::Vec2f>(i, j)[1] = (-u*P.at<cv::Vec2f>(i, j)[0] - v*Q.at<cv::Vec2f>(i, j)[0]) / d;
            }
        }
    }
    
    /* setting unknown average height to zero */
    Z.at<cv::Vec2f>(0, 0)[0] = 0;
    Z.at<cv::Vec2f>(0, 0)[1] = 0;
    
    cv::dft(Z, Z, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    
    return Z;
}

int main() {
    
    /* reading images */
    cv::Mat img1 = cv::imread("../../images/bunny_1.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat img2 = cv::imread("../../images/bunny_2.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat img3 = cv::imread("../../images/bunny_3.png", CV_LOAD_IMAGE_GRAYSCALE);
    
    /* light directions, surface normals, p,q gradients */
    cv::Mat Lights = (cv::Mat_<float>(3,3) << -0.2, 0.0, 1.0, 0.2, -0.2, 1.0, 0.2, 0.2, 1.0);
    cv::Mat Normals(img1.rows, img1.cols, CV_32FC3, cv::Scalar::all(0));
    cv::Mat Pgrads(img1.rows, img1.cols, CV_32F, cv::Scalar::all(0));
    cv::Mat Qgrads(img1.rows, img1.cols, CV_32F, cv::Scalar::all(0));
    
    /* estimate surface normals and p,q gradients */
    for (int y=0; y<img1.rows; y++) {
        for (int x=0; x<img1.cols; x++) {
            cv::Vec3f I(img1.at<uchar>(y,x),
                        img2.at<uchar>(y,x),
                        img3.at<uchar>(y,x));
            cv::Mat n = Lights.inv() * cv::Mat(I);
            float p = sqrt(cv::Mat(n).dot(n));
            if (p > 0) { n = n/p; }
            if (n.at<float>(2,0) == 0) { n.at<float>(2,0) = 1.0; }
            Normals.at<cv::Vec3f>(cv::Point(x,y)) = n;
            Pgrads.at<float>(cv::Point(x,y)) = -n.at<float>(0,0);
            Qgrads.at<float>(cv::Point(x,y)) = n.at<float>(1,0);
        }
    }
    
    /* global integration of surface normals */
    cv::Mat Z = globalHeights(Pgrads, Qgrads);
    
    /* display reconstruction */
    displayMesh(Pgrads.cols, Pgrads.rows, Z);

    return 0;
}