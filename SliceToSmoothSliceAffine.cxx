#include "itkExtractImageFilter.h"
#include "itkImage.h"
#include "itkImageRegistrationMethodv4.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkMeanSquaresImageToImageMetricv4.h"
#include "itkGradientDescentOptimizer.h"
#include "itkLBFGSOptimizerv4.h"
#include "itkResampleImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkCompositeTransform.h"
#include "itkRescaleIntensityImageFilter.h"
#include <itkSmoothingRecursiveGaussianImageFilter.h>
#include "itkAffineTransform.h"
#include <itkConjugateGradientOptimizer.h>

#include <itkJoinSeriesImageFilter.h>

#include "ImageIO.h"

#include <tclap/CmdLine.h>

typedef  float  PixelType;
 
typedef itk::Image< PixelType, 2 >  ImageType2D;
typedef itk::Image< PixelType, 3 >  ImageType3D;
     
typedef itk::SmoothingRecursiveGaussianImageFilter<ImageType3D, ImageType3D> GaussianFilter3D;
typedef itk::RescaleIntensityImageFilter<ImageType3D, ImageType3D> RescaleFilter;
typedef itk::ExtractImageFilter< ImageType3D, ImageType2D > ExtractFilter;
typedef itk::JoinSeriesImageFilter< ImageType2D, ImageType3D> JoinFilter;
typedef itk::ResampleImageFilter< ImageType2D, ImageType2D >    ResampleFilterType;
 

typedef itk::CompositeTransform< double, 2 > CompositeTransform;

//typedef itk::GradientDescentOptimizer       OptimizerType;
//typedef itk::ConjugateGradientOptimizer       OptimizerType;
typedef itk::LBFGSOptimizerv4       OptimizerType;

//2D registration
typedef itk::MeanSquaresImageToImageMetricv4< ImageType2D, ImageType2D >  MetricType2D;
typedef itk::LinearInterpolateImageFunction< ImageType2D, double >    InterpolatorType2D;
typedef itk::ImageRegistrationMethodv4< ImageType2D, ImageType2D >    RegistrationType2D;
typedef itk::AffineTransform< double, 2 >     TransformType2D;
typedef TransformType2D::ParametersType     ParametersType2D;


int main(int argc, char **argv ){

  //Command line parsing
  TCLAP::CmdLine cmd("BSpline registration of 3D ultrasound bubble image", ' ', "1");

  TCLAP::ValueArg<std::string> volumeArg("v","volume","Ultrasound volume image", true, "",
      "filename");
  cmd.add(volumeArg);

  TCLAP::ValueArg< int > dimArg("d","dim","Along which dimension to register", false, 2,
      "integer");
  cmd.add( dimArg );

  TCLAP::ValueArg<float> smooth1Arg("","smooth1",
      "Pre smooth volume with a Gaussian"
      , false, 6.0, "float");
  cmd.add(smooth1Arg);
  
  TCLAP::ValueArg<float> smooth2Arg("","smooth2",
      "Pre smooth volume with a Gaussian"
      , false, 6.0, "float");
  cmd.add(smooth2Arg);
  
  TCLAP::ValueArg<float> smooth3Arg("","smooth3",
      "Pre smooth volume with a Gaussian"
      , false, 6.0, "float");
  cmd.add(smooth3Arg);
  
  TCLAP::ValueArg<float> smooth3MArg("","smooth3M",
      "Pre smooth volume with a Gaussian"
      , false, 0.5, "float");
  cmd.add(smooth3MArg);


  TCLAP::ValueArg<std::string> prefixArg("p","prefix","Prefix for storing output images", true, "",
      "filename");
  cmd.add(prefixArg);
  
  
  try{
    cmd.parse( argc, argv );
  } 
  catch (TCLAP::ArgException &e){ 
    std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; 
    return -1;
  }

  std::string prefix = prefixArg.getValue();
 
  int dim = dimArg.getValue();
  
  double s1 = smooth1Arg.getValue();
  double s2 = smooth2Arg.getValue();
  double s3 = smooth3Arg.getValue();
  double s3M = smooth3MArg.getValue();

  // Get the  volume
  ImageType3D::Pointer volume = ImageIO<ImageType3D>::readImage( volumeArg.getValue() );      
  RescaleFilter::Pointer rescale = RescaleFilter::New();
  rescale->SetInput(volume);
  rescale->SetOutputMaximum(100.0);
  rescale->SetOutputMinimum(0.0);
  rescale->Update();
  volume = rescale->GetOutput();

  ImageType3D::SpacingType volumeSpacing = volume->GetSpacing();

  ImageType3D::RegionType volumeRegion = volume->GetLargestPossibleRegion();
  ImageType3D::SizeType volumeSize = volumeRegion.GetSize();

  //Smooth s3 should be on the order of the breathing period
  GaussianFilter3D::SigmaArrayType sig = GaussianFilter3D::SigmaArrayType::Filled(0);
  sig[0] = s1 * volumeSpacing[0]; 
  sig[1] = s2 * volumeSpacing[1]; 
  sig[2] = s3 * volumeSpacing[2]; 
  
  GaussianFilter3D::Pointer smoothOrig = GaussianFilter3D::New();
  smoothOrig->SetSigmaArray(sig);
  smoothOrig->SetInput( volume );
  smoothOrig->Update();
  ImageType3D::Pointer volumeSmooth = smoothOrig->GetOutput();
   
  { 
  std::stringstream outfile;
  outfile << prefix << "-smoothssa-smooth.tif";
  ImageIO<ImageType3D>::saveImage( volumeSmooth, outfile.str()  );
  }
  
  sig[2] = s3M * volumeSpacing[2]; 
  
  GaussianFilter3D::Pointer smoothMoving = GaussianFilter3D::New();
  smoothMoving->SetSigmaArray(sig);
  smoothMoving->SetInput( volume );
  smoothMoving->Update();
  ImageType3D::Pointer volumeMoving = smoothMoving->GetOutput();


  //Join all the registred slices
  JoinFilter::Pointer joinFilter = JoinFilter::New();

  //Step 1: Register slice to time smoothed slice 
  for(int i=0; i < volumeSize[dim]; i++){



    std::cout << "registering slice: " << i << std::endl;
    
    //Extract consecutive slices
    ImageType3D::IndexType fixedStart;
    fixedStart.Fill(0);
    fixedStart[dim] = i;
    
    ImageType3D::IndexType movingStart;
    movingStart.Fill(0);
    movingStart[dim] = i;
 
    ImageType3D::SizeType size = volumeSize;
    size[dim] = 0;

 
    //GaussianFilter2D::SigmaArrayType sig2d = GaussianFilter2D::SigmaArrayType::Filled(0);
    //sig2d[0] = s1 * volumeSpacing[0]; 
    //sig2d[1] = s2 * volumeSpacing[1]; 
 
    ImageType3D::RegionType fixedRegion(fixedStart, size);
    ExtractFilter::Pointer fixedExtract = ExtractFilter::New();
    fixedExtract->SetExtractionRegion(fixedRegion);
    fixedExtract->SetInput(volumeSmooth);
    fixedExtract->SetDirectionCollapseToIdentity(); // This is required.
    fixedExtract->Update();
    ImageType2D::Pointer fixedOrig = fixedExtract->GetOutput();
    //GaussianFilter2D::Pointer smoothFixed = GaussianFilter2D::New();
    //smoothFixed->SetSigmaArray(sig2d);
    //smoothFixed->SetInput( fixedExtract->GetOutput() );
    //smoothFixed->Update();
    ImageType2D::Pointer fixedImage = fixedOrig; //smoothFixed->GetOutput();

    
    ImageType3D::RegionType movingRegion(movingStart, size);
    ExtractFilter::Pointer movingExtract = ExtractFilter::New();
    movingExtract->SetExtractionRegion(movingRegion);
    movingExtract->SetInput(volume);
    movingExtract->SetDirectionCollapseToIdentity(); // This is required.
    movingExtract->Update();
    ImageType2D::Pointer movingOrig = movingExtract->GetOutput();


    ExtractFilter::Pointer movingExtract2 = ExtractFilter::New();
    movingExtract2->SetExtractionRegion(movingRegion);
    movingExtract2->SetInput(volumeMoving);
    movingExtract2->SetDirectionCollapseToIdentity(); // This is required.
    movingExtract2->Update();
    ImageType2D::Pointer movingImage = movingExtract2->GetOutput();
/*
    GaussianFilter2D::Pointer smoothMoving = GaussianFilter2D::New();
    smoothMoving->SetSigmaArray(sig2d);
    smoothMoving->SetInput( movingExtract->GetOutput() );
    smoothMoving->Update();
    ImageType2D::Pointer movingImage = smoothMoving->GetOutput();
*/



    //Setup bspline deformation
    TransformType2D::Pointer      transform     = TransformType2D::New();

   
    //Setup registration 
    MetricType2D::Pointer         metric        = MetricType2D::New();
    OptimizerType::Pointer      optimizer       = OptimizerType::New();
    InterpolatorType2D::Pointer   movingInterpolator  = InterpolatorType2D::New();
    InterpolatorType2D::Pointer   fixedInterpolator  = InterpolatorType2D::New();
    RegistrationType2D::Pointer   registration  = RegistrationType2D::New();


    OptimizerType::ScalesType scales( transform->GetNumberOfParameters() );
    scales[0] = 50.0;
    scales[1] = 50.0;
    scales[2] = 50.0; 
    scales[3] = 50.0; 
    scales[4] = 1.0; 
    scales[5] = 1.0; 
 
    optimizer->SetScales( scales );
    
    registration->SetMetric(        metric        );
    registration->SetOptimizer(     optimizer     );
    //registration->SetTransform(     transform     );
    metric->SetMovingInterpolator( movingInterpolator );
    metric->SetFixedInterpolator( fixedInterpolator );
    //registration->SetInterpolator(  interpolator  );


    registration->SetFixedImage(fixedImage);
    registration->SetMovingImage(movingImage);
 
    //registration->SetFixedImageRegion( fixedImage->GetLargestPossibleRegion() );
 
 
    std::cout <<  transform->GetParameters()  << std::endl;
    registration->SetInitialTransform( transform );



    //Setup optimizer 
    
    //optimizer->SetMaximumStepLength( .1 ); 
    //optimizer->SetMinimumStepLength( 0.0001 );
    
     //optimizer->SetLearningRate( 0.5 );
     //optimizer->SetNumberOfIterations( 200 );
     
    optimizer->SetGradientConvergenceTolerance( 0.000001 );
    optimizer->SetLineSearchAccuracy( 0.5 );
    optimizer->SetDefaultStepLength( 0.001 );
    optimizer->TraceOn();
    optimizer->SetMaximumNumberOfFunctionEvaluations( 200 );

    //Do registration
    try{
      registration->SetNumberOfThreads(4);
      registration->Update();
    }
    catch( itk::ExceptionObject & err ){
      std::cerr << "ExceptionObject caught !" << std::endl;
      std::cerr << err << std::endl;
      //return EXIT_FAILURE;
    }
 
    const double bestValue = optimizer->GetValue();
    std::cout << "Result = " << std::endl;
    std::cout << " Metric value  = " << bestValue          << std::endl;
 

    //Move the orginal moving slice image into the frame of slice 0 
    ResampleFilterType::Pointer resampler = ResampleFilterType::New();
    resampler->SetInput( movingOrig );
    resampler->SetTransform( registration->GetTransform() );
    resampler->SetSize( fixedImage->GetLargestPossibleRegion().GetSize() );
    resampler->SetOutputOrigin(  fixedImage->GetOrigin() );
    resampler->SetOutputSpacing( fixedImage->GetSpacing() );
    resampler->SetOutputDirection( fixedImage->GetDirection() );
    resampler->SetDefaultPixelValue( 0 );
    resampler->Update();
    ImageType2D::Pointer moved = resampler->GetOutput();
	

    //Join all the images nto a volume
    joinFilter->SetInput(i, moved); 

    std::cout <<  transform->GetParameters()  << std::endl;
  }

  joinFilter->Update();
  ImageType3D::Pointer alignedVolume = joinFilter->GetOutput();
  
  { 
  std::stringstream outfile;
  outfile << prefix << "-smoothsa-registered.tif";
  ImageIO<ImageType3D>::saveImage( alignedVolume, outfile.str()  );
  }





  return EXIT_SUCCESS;
}
