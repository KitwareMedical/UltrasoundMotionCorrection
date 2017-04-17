#include "itkExtractImageFilter.h"
#include "itkImage.h"
#include "itkImageRegistrationMethod.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkGradientDescentOptimizer.h"
#include "itkLBFGSOptimizer.h"
#include "itkResampleImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkCompositeTransform.h"
#include "itkRescaleIntensityImageFilter.h"
#include <itkSmoothingRecursiveGaussianImageFilter.h>
#if ITK_VERSION_MAJOR < 4
#include "itkBSplineDeformableTransform.h"
#else
#include "itkBSplineTransform.h"
#endif
#include <itkConjugateGradientOptimizer.h>

#include <itkJoinSeriesImageFilter.h>

#include "ImageIO.h"

#include <tclap/CmdLine.h>

typedef  float  PixelType;
 
typedef itk::Image< PixelType, 3 >  ImageType3D;
     
typedef itk::SmoothingRecursiveGaussianImageFilter<ImageType3D, ImageType3D> GaussianFilter3D;
typedef itk::RescaleIntensityImageFilter<ImageType3D, ImageType3D> RescaleFilter;
typedef itk::ResampleImageFilter< ImageType3D, ImageType3D >    ResampleFilterType3D;
 

typedef itk::CompositeTransform< double, 2 > CompositeTransform;

//typedef itk::GradientDescentOptimizer       OptimizerType;
//typedef itk::ConjugateGradientOptimizer       OptimizerType;
typedef itk::LBFGSOptimizer       OptimizerType;


//3d Registration
typedef itk::MeanSquaresImageToImageMetric< ImageType3D, ImageType3D >  MetricType3D;
typedef itk::LinearInterpolateImageFunction< ImageType3D, double >    InterpolatorType3D;
typedef itk::ImageRegistrationMethod< ImageType3D, ImageType3D >    RegistrationType3D;

typedef itk::BSplineTransform< double, 3,  2 >     TransformType3D;
typedef TransformType3D::ParametersType     ParametersType3D;


int main(int argc, char **argv ){

  //Command line parsing
  TCLAP::CmdLine cmd("BSpline registration of 3D ultrasound bubble image", ' ', "1");

  TCLAP::ValueArg<std::string> volumeArg("v","volume","Ultrasound volume image", true, "",
      "filename");
  cmd.add(volumeArg);

  TCLAP::ValueArg< int > dimArg("d","dim","Along which dimension to register", false, 2,
      "integer");
  cmd.add( dimArg );

  TCLAP::ValueArg<float> smooth1Arg("","smooth1F",
      "Pre smooth volume with a Gaussian"
      , false, 3.0, "float");
  cmd.add(smooth1Arg);
  
  TCLAP::ValueArg<float> smooth2Arg("","smooth2F",
      "Smooth fixed volume with a Gaussian"
      , false, 3.0, "float");
  cmd.add(smooth2Arg);
  
  TCLAP::ValueArg<float> smooth3Arg("","smooth3F",
      "Smooth fixed volume with a Gaussian"
      , false, 8.0, "float");
  cmd.add(smooth3Arg);
  
  TCLAP::ValueArg<float> smooth1MArg("","smooth1M",
      "Smooth fixed volume with a Gaussian"
      , false, 6.0, "float");
  cmd.add(smooth1MArg);
  
  TCLAP::ValueArg<float> smooth2MArg("","smooth2M",
      "Smooth moving volume with a Gaussian"
      , false, 6.0, "float");
  cmd.add(smooth2MArg);
  
  TCLAP::ValueArg<float> smooth3MArg("","smooth3M",
      "Smooth moving volume with a Gaussian"
      , false, 0.5, "float");
  cmd.add(smooth3MArg);
  
  TCLAP::ValueArg< int > gridXArg("","gridX","Number of grid points in X for Bspline", false, 3,
      "integer");
  cmd.add( gridXArg );

  TCLAP::ValueArg< int > gridYArg("","gridY","Number of grid points in Y for Bspline", false, 6,
      "integer");
  cmd.add( gridYArg );

  TCLAP::ValueArg< int > gridTArg("","gridT","Number of grid points in time for Bspline", false, 30,
      "integer");
  cmd.add( gridTArg );

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
 
  double s1F = smooth1Arg.getValue();
  double s2F = smooth2Arg.getValue();
  double s3F = smooth3Arg.getValue();
  
  double s1M = smooth1MArg.getValue();
  double s2M = smooth2MArg.getValue();
  double s3M = smooth3MArg.getValue();

  // Get the  volume
  ImageType3D::Pointer volume = ImageIO<ImageType3D>::readImage( volumeArg.getValue() );      
  RescaleFilter::Pointer rescale = RescaleFilter::New();
  rescale->SetInput(volume);
  rescale->SetOutputMaximum(10.0);
  rescale->SetOutputMinimum(0.0);
  rescale->Update();
  volume = rescale->GetOutput();

  ImageType3D::SpacingType volumeSpacing = volume->GetSpacing();
  ImageType3D::RegionType volumeRegion = volume->GetLargestPossibleRegion();
  ImageType3D::SizeType volumeSize = volumeRegion.GetSize();
  


  GaussianFilter3D::SigmaArrayType sigF = GaussianFilter3D::SigmaArrayType::Filled(0);
  sigF[0] = s1F * volumeSpacing[0]; 
  sigF[1] = s2F * volumeSpacing[1]; 
  sigF[2] = s3F * volumeSpacing[2];  
  GaussianFilter3D::Pointer smoothFixed = GaussianFilter3D::New();
  smoothFixed->SetSigmaArray(sigF);
  smoothFixed->SetInput( volume );
  smoothFixed->Update();
  ImageType3D::Pointer volumeFixed = smoothFixed->GetOutput();
  	
  { 
  std::stringstream outfile;
  outfile << prefix << "-vbsp-fixed.nrrd";
  ImageIO<ImageType3D>::saveImage( volumeFixed, outfile.str()  );
  }




  GaussianFilter3D::SigmaArrayType sigM = GaussianFilter3D::SigmaArrayType::Filled(0);
  sigM[0] = s1M * volumeSpacing[0]; 
  sigM[1] = s2M * volumeSpacing[1]; 
  sigM[2] = s3M * volumeSpacing[2]; 
  GaussianFilter3D::Pointer smoothMoving = GaussianFilter3D::New();
  smoothMoving->SetSigmaArray(sigM);
  smoothMoving->SetInput( volume );
  smoothMoving->Update();
  ImageType3D::Pointer volumeMoving = smoothMoving->GetOutput();
  	
  { 
  std::stringstream outfile;
  outfile << prefix << "-vbsp-moving.nrrd";
  ImageIO<ImageType3D>::saveImage( volumeMoving, outfile.str()  );
  }

  //Setup bspline deformation
  TransformType3D::Pointer      transform3D     = TransformType3D::New();

  TransformType3D::PhysicalDimensionsType   fixedPhysicalDimensions3D;
  TransformType3D::MeshSizeType             meshSize3D;
  for( unsigned int j=0; j < 3; j++ ){
      fixedPhysicalDimensions3D[j] = volume->GetSpacing()[j] *
		    static_cast<double>(volume->GetLargestPossibleRegion().GetSize()[j] - 1 );
  }
  meshSize3D[0] = gridXArg.getValue();
  meshSize3D[1] = gridYArg.getValue();
  meshSize3D[2] = gridTArg.getValue();
    
  transform3D->SetTransformDomainOrigin( volume->GetOrigin() );
  transform3D->SetTransformDomainPhysicalDimensions( fixedPhysicalDimensions3D );
  transform3D->SetTransformDomainMeshSize( meshSize3D );
  transform3D->SetTransformDomainDirection( volume->GetDirection() );

  //Setup registration 
  MetricType3D::Pointer         metric3D        = MetricType3D::New();
  OptimizerType::Pointer        optimizer3D     = OptimizerType::New();
  InterpolatorType3D::Pointer   interpolator3D  = InterpolatorType3D::New();
  RegistrationType3D::Pointer   registration3D  = RegistrationType3D::New();


  registration3D->SetMetric(        metric3D        );
  registration3D->SetOptimizer(     optimizer3D     );
  registration3D->SetTransform(     transform3D     );
  registration3D->SetInterpolator(  interpolator3D  );


  registration3D->SetFixedImage( volumeFixed );
  registration3D->SetMovingImage( volumeMoving );
  registration3D->SetFixedImageRegion( volume->GetLargestPossibleRegion() );


  const unsigned int numberOfParameters = transform3D->GetNumberOfParameters();
  ParametersType3D parameters( numberOfParameters );
  parameters.Fill( 0.0 );
  transform3D->SetParameters( parameters );
  registration3D->SetInitialTransformParameters( transform3D->GetParameters() );



  //Setup optimizer 

  //optimizer->SetMaximumStepLength( .1 ); 
  //optimizer->SetMinimumStepLength( 0.0001 );

  //optimizer->SetLearningRate( 0.5 );
  //optimizer->SetNumberOfIterations( 200 );

  optimizer3D->SetGradientConvergenceTolerance( 0.000001 );
  optimizer3D->SetLineSearchAccuracy( 0.5 );
  optimizer3D->SetDefaultStepLength( 0.1 );
  optimizer3D->TraceOn();
  optimizer3D->SetMaximumNumberOfFunctionEvaluations( 200 );

  //Do registration
  std::cout << "starting 3D registration" << std::endl;
  try{
	  registration3D->SetNumberOfThreads(1);
	  registration3D->Update();
  }
  catch( itk::ExceptionObject & err ){
	  std::cerr << "ExceptionObject caught !" << std::endl;
	  std::cerr << err << std::endl;
	  return EXIT_FAILURE;
  }

  const double bestValue = optimizer3D->GetValue();
  std::cout << "Result = " << std::endl;
  std::cout << " Metric value  = " << bestValue          << std::endl;
    
  //Move the orginal moving slice image into the frame of slice 0 
  ResampleFilterType3D::Pointer resampler = ResampleFilterType3D::New();
  resampler->SetInput( volume );
  resampler->SetTransform( registration3D->GetTransform() );
  resampler->SetSize( volume->GetLargestPossibleRegion().GetSize() );
  resampler->SetOutputOrigin(  volume->GetOrigin() );
  resampler->SetOutputSpacing( volume->GetSpacing() );
  resampler->SetOutputDirection( volume->GetDirection() );
  resampler->SetDefaultPixelValue( 0 );
  resampler->Update();
  ImageType3D::Pointer moved = resampler->GetOutput();
	
  { 
  std::stringstream outfile;
  outfile << prefix << "-vbsp-registered.nrrd";
  ImageIO<ImageType3D>::saveImage( moved, outfile.str()  );
  }



  return EXIT_SUCCESS;
}
