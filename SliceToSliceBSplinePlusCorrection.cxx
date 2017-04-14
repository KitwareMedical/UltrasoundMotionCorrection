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
 
typedef itk::Image< PixelType, 2 >  ImageType2D;
typedef itk::Image< PixelType, 3 >  ImageType3D;
     
typedef itk::SmoothingRecursiveGaussianImageFilter<ImageType3D, ImageType3D> GaussianFilter3D;
typedef itk::SmoothingRecursiveGaussianImageFilter<ImageType2D, ImageType2D> GaussianFilter2D;
typedef itk::RescaleIntensityImageFilter<ImageType3D, ImageType3D> RescaleFilter;
typedef itk::ExtractImageFilter< ImageType3D, ImageType2D > ExtractFilter;
typedef itk::JoinSeriesImageFilter< ImageType2D, ImageType3D> JoinFilter;
typedef itk::ResampleImageFilter< ImageType2D, ImageType2D >    ResampleFilterType;
typedef itk::ResampleImageFilter< ImageType3D, ImageType3D >    ResampleFilterType3D;
 

typedef itk::CompositeTransform< double, 2 > CompositeTransform;

//typedef itk::GradientDescentOptimizer       OptimizerType;
//typedef itk::ConjugateGradientOptimizer       OptimizerType;
typedef itk::LBFGSOptimizer       OptimizerType;

//2D registration
typedef itk::MeanSquaresImageToImageMetric< ImageType2D, ImageType2D >  MetricType2D;
typedef itk::LinearInterpolateImageFunction< ImageType2D, double >    InterpolatorType2D;
typedef itk::ImageRegistrationMethod< ImageType2D, ImageType2D >    RegistrationType2D;

typedef itk::BSplineTransform< double, 2,  2 >     TransformType2D;
typedef TransformType2D::ParametersType     ParametersType2D;

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

  TCLAP::ValueArg<float> smooth1Arg("","smooth1",
      "Pre smooth volume with a Gaussian"
      , false, 1.0, "float");
  cmd.add(smooth1Arg);
  
  TCLAP::ValueArg<float> smooth2Arg("","smooth2",
      "Pre smooth volume with a Gaussian"
      , false, 1.0, "float");
  cmd.add(smooth2Arg);
  
  TCLAP::ValueArg<float> smooth3Arg("","smooth3",
      "Pre smooth volume with a Gaussian"
      , false, 1.0, "float");
  cmd.add(smooth3Arg);

  TCLAP::ValueArg< int > gridXArg("","gridX","Number of grid points in X for Bspline", false, 6,
      "integer");
  cmd.add( gridXArg );

  TCLAP::ValueArg< int > gridYArg("","gridY","Number of grid points in Y for Bspline", false, 6,
      "integer");
  cmd.add( gridYArg );


  TCLAP::ValueArg< int > gridTArg("","gridT","Number of grid points in time for Bspline", false, 6,
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
 
  int dim = dimArg.getValue();
  
  double s1 = smooth1Arg.getValue();
  double s2 = smooth2Arg.getValue();
  double s3 = smooth3Arg.getValue();

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


  //Store registration to register to slice 0
  std::list< RegistrationType2D::TransformPointer > transforms;
  //Join all the registred slices
  JoinFilter::Pointer joinFilter = JoinFilter::New();

  //Step 1: Register slice to consecutive slice 
  for(int i=1; i < volumeSize[dim]; i++){



    std::cout << "registering slice: " << i << std::endl;
    
    //Extract consecutive slices
    ImageType3D::IndexType fixedStart;
    fixedStart.Fill(0);
    fixedStart[dim] = i-1;
    
    ImageType3D::IndexType movingStart;
    movingStart.Fill(0);
    movingStart[dim] = i;
 
    ImageType3D::SizeType size = volumeSize;
    size[dim] = 0;

 
    GaussianFilter2D::SigmaArrayType sig2d = GaussianFilter2D::SigmaArrayType::Filled(0);
    sig2d[0] = s1 * volumeSpacing[0]; 
    sig2d[1] = s2 * volumeSpacing[1]; 
 
    ImageType3D::RegionType fixedRegion(fixedStart, size);
    ExtractFilter::Pointer fixedExtract = ExtractFilter::New();
    fixedExtract->SetExtractionRegion(fixedRegion);
    fixedExtract->SetInput(volume);
    fixedExtract->SetDirectionCollapseToIdentity(); // This is required.
    fixedExtract->Update();
    ImageType2D::Pointer fixedOrig = fixedExtract->GetOutput();
    GaussianFilter2D::Pointer smoothFixed = GaussianFilter2D::New();
    smoothFixed->SetSigmaArray(sig2d);
    smoothFixed->SetInput( fixedExtract->GetOutput() );
    smoothFixed->Update();
    ImageType2D::Pointer fixedImage = smoothFixed->GetOutput();

    
    ImageType3D::RegionType movingRegion(movingStart, size);
    ExtractFilter::Pointer movingExtract = ExtractFilter::New();
    movingExtract->SetExtractionRegion(movingRegion);
    movingExtract->SetInput(volume);
    movingExtract->SetDirectionCollapseToIdentity(); // This is required.
    movingExtract->Update();
    ImageType2D::Pointer movingOrig = movingExtract->GetOutput();
    GaussianFilter2D::Pointer smoothMoving = GaussianFilter2D::New();
    smoothMoving->SetSigmaArray(sig2d);
    smoothMoving->SetInput( movingExtract->GetOutput() );
    smoothMoving->Update();
    ImageType2D::Pointer movingImage = smoothMoving->GetOutput();




    //Setup bspline deformation
    TransformType2D::Pointer      transform     = TransformType2D::New();

    TransformType2D::PhysicalDimensionsType   fixedPhysicalDimensions;
    TransformType2D::MeshSizeType             meshSize;
    for( unsigned int j=0; j < 2; j++ ){
	    fixedPhysicalDimensions[j] = fixedImage->GetSpacing()[j] *
		    static_cast<double>(fixedImage->GetLargestPossibleRegion().GetSize()[j] - 1 );
    }
    meshSize[0] = gridXArg.getValue();
    meshSize[1] = gridYArg.getValue();
    transform->SetTransformDomainOrigin( fixedImage->GetOrigin() );
    transform->SetTransformDomainPhysicalDimensions( fixedPhysicalDimensions );
    transform->SetTransformDomainMeshSize( meshSize );
    transform->SetTransformDomainDirection( fixedImage->GetDirection() );

   
    //Setup registration 
    MetricType2D::Pointer         metric        = MetricType2D::New();
    OptimizerType::Pointer      optimizer       = OptimizerType::New();
    InterpolatorType2D::Pointer   interpolator  = InterpolatorType2D::New();
    RegistrationType2D::Pointer   registration  = RegistrationType2D::New();


    registration->SetMetric(        metric        );
    registration->SetOptimizer(     optimizer     );
    registration->SetTransform(     transform     );
    registration->SetInterpolator(  interpolator  );


    registration->SetFixedImage(fixedImage);
    registration->SetMovingImage(movingImage);
 
    registration->SetFixedImageRegion( fixedImage->GetLargestPossibleRegion() );
 
 
    const unsigned int numberOfParameters = transform->GetNumberOfParameters();
    ParametersType2D parameters( numberOfParameters );
    parameters.Fill( 0.0 );
    transform->SetParameters( parameters );
    registration->SetInitialTransformParameters( transform->GetParameters() );



    //Setup optimizer 
    
    //optimizer->SetMaximumStepLength( .1 ); 
    //optimizer->SetMinimumStepLength( 0.0001 );
    
     //optimizer->SetLearningRate( 0.5 );
     //optimizer->SetNumberOfIterations( 200 );
      
    optimizer->SetGradientConvergenceTolerance( 0.000001 );
    optimizer->SetLineSearchAccuracy( 0.5 );
    optimizer->SetDefaultStepLength( 0.1 );
    optimizer->TraceOn();
    optimizer->SetMaximumNumberOfFunctionEvaluations( 200 );

    //Do registration
    try{
      registration->SetNumberOfThreads(1);
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
 
    //Add the transfrom to the list of transfroms
    transforms.push_back( registration->GetTransform() );

    //Build composite transform
    CompositeTransform::Pointer composite = CompositeTransform::New();
    for( std::list< RegistrationType2D::TransformPointer >::iterator it = transforms.begin();
		    it != transforms.end(); ++it){
      composite->AddTransform( *it );
    }



    //Move the orginal moving slice image into the frame of slice 0 
    ResampleFilterType::Pointer resampler = ResampleFilterType::New();
    resampler->SetInput( movingOrig );
    resampler->SetTransform( composite );
    resampler->SetSize( fixedImage->GetLargestPossibleRegion().GetSize() );
    resampler->SetOutputOrigin(  fixedImage->GetOrigin() );
    resampler->SetOutputSpacing( fixedImage->GetSpacing() );
    resampler->SetOutputDirection( fixedImage->GetDirection() );
    resampler->SetDefaultPixelValue( 0 );
    resampler->Update();
    ImageType2D::Pointer moved = resampler->GetOutput();
	

    //Join all the images nto a volume
    joinFilter->SetInput(i, moved); 

    if( i== 1){
      joinFilter->SetInput(0, fixedOrig); 
    }
  }

  joinFilter->Update();
  ImageType3D::Pointer alignedVolume = joinFilter->GetOutput();

  { 
  std::stringstream outfile;
  outfile << prefix << "-ssbsc-slice-registered.nrrd";
  ImageIO<ImageType3D>::saveImage( alignedVolume, outfile.str()  );
  }









  std::cout << "Step 2: registering volumes" << std::endl;
  //Step 2: register aligned volume and original volume
  GaussianFilter3D::SigmaArrayType sig = GaussianFilter3D::SigmaArrayType::Filled(0);
  sig[0] = s1 * volumeSpacing[0]; 
  sig[1] = s2 * volumeSpacing[1]; 
  sig[2] = s3 * volumeSpacing[2]; 
  
  GaussianFilter3D::Pointer smoothOrig = GaussianFilter3D::New();
  smoothOrig->SetSigmaArray(sig);
  smoothOrig->SetInput( volume );
  smoothOrig->Update();
  ImageType3D::Pointer volumeSmooth = smoothOrig->GetOutput();

  GaussianFilter3D::Pointer smoothAlign = GaussianFilter3D::New();
  smoothAlign->SetSigmaArray(sig);
  smoothAlign->SetInput( alignedVolume );
  smoothAlign->Update();
  ImageType3D::Pointer alignedVolumeSmooth = smoothAlign->GetOutput();
  





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


  registration3D->SetFixedImage(volumeSmooth);
  registration3D->SetMovingImage( alignedVolumeSmooth );

  registration3D->SetFixedImageRegion( volumeSmooth->GetLargestPossibleRegion() );


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
  resampler->SetInput( alignedVolume );
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
  outfile << prefix << "-ssbsc-registered.nrrd";
  ImageIO<ImageType3D>::saveImage( moved, outfile.str()  );
  }



  return EXIT_SUCCESS;
}
