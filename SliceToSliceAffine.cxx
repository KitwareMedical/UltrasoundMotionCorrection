#include "itkExtractImageFilter.h"
#include "itkImage.h"
#include "itkImageRegistrationMethod.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkGradientDescentOptimizer.h"
#include "itkLBFGSOptimizer.h"
#include "itkResampleImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkSpatialObjectToImageFilter.h"
#include "itkAffineTransform.h"
#include "itkTranslationTransform.h"
#include "itkCompositeTransform.h"
#include "itkRescaleIntensityImageFilter.h"
#include <itkSmoothingRecursiveGaussianImageFilter.h>
#include <itkConjugateGradientOptimizer.h>
#include <itkJoinSeriesImageFilter.h>

#include "ImageIO.h"

#include <tclap/CmdLine.h>

typedef  float  PixelType;
 
typedef itk::Image< PixelType, 2 >  ImageType2D;
typedef itk::Image< PixelType, 3 >  ImageType3D;
     
typedef itk::SmoothingRecursiveGaussianImageFilter<ImageType2D, ImageType2D> GaussianFilter2D;
typedef itk::ExtractImageFilter< ImageType3D, ImageType2D > ExtractFilter;
typedef itk::RescaleIntensityImageFilter<ImageType3D, ImageType3D> RescaleFilter;
typedef itk::JoinSeriesImageFilter< ImageType2D, ImageType3D> JoinFilter;



int main(int argc, char **argv )
{

  //Command line parsing
  TCLAP::CmdLine cmd("Affine registration of 3D ultrasound bubble image", ' ', "1");

  TCLAP::ValueArg<std::string> volumeArg("v","volume","Ultrasound volume image", true, "",
      "filename");
  cmd.add(volumeArg);

  TCLAP::ValueArg< int > dimArg("d","dim","Along whihc dimension to register", false, 2,
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


  TCLAP::ValueArg<std::string> prefixArg("p","prefix","Prefix for stroing output images", true, "",
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

  //List for strong trsanfroms between slices
  typedef itk::ImageRegistrationMethod< ImageType2D, ImageType2D >    RegistrationType;
  std::list< RegistrationType::TransformPointer > transforms;
  

  JoinFilter::Pointer joinFilter = JoinFilter::New();
  int dim = dimArg.getValue();
  //Between each slice in the volume compute an affine transfrom
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



    

    //Setup Affine transfrom registratiom
    typedef itk::TranslationTransform< double, 2> TransformType;
    //typedef itk::GradientDescentOptimizer       OptimizerType;
    //typedef itk::ConjugateGradientOptimizer OptimizerType;
    typedef itk::LBFGSOptimizer       OptimizerType;
    typedef itk::MeanSquaresImageToImageMetric< ImageType2D, ImageType2D >  MetricType;
    typedef itk::LinearInterpolateImageFunction< ImageType2D, double >    InterpolatorType;
 
        
    MetricType::Pointer         metric        = MetricType::New();
    TransformType::Pointer      transform     = TransformType::New();
    OptimizerType::Pointer      optimizer     = OptimizerType::New();
    InterpolatorType::Pointer   interpolator  = InterpolatorType::New();
    RegistrationType::Pointer   registration  = RegistrationType::New();
 
    registration->SetMetric(        metric        );
    registration->SetOptimizer(     optimizer     );
    registration->SetTransform(     transform     );
    registration->SetInterpolator(  interpolator  );


    registration->SetFixedImage(fixedImage);
    registration->SetMovingImage(movingImage);
    registration->SetFixedImageRegion( fixedImage->GetLargestPossibleRegion() );
    registration->SetInitialTransformParameters( transform->GetParameters() );
 
    //optimizer->SetMaximumStepLength( .1 ); 
    //optimizer->SetMinimumStepLength( 0.0001 );
    
    //optimizer->SetLearningRate( 0.001 );
    //optimizer->SetNumberOfIterations( 200 );
       
    optimizer->SetGradientConvergenceTolerance( 0.00001 );
    optimizer->SetLineSearchAccuracy( 0.5 );
    optimizer->SetDefaultStepLength( 0.01 );
    optimizer->TraceOn();
    optimizer->SetMaximumNumberOfFunctionEvaluations( 100 );


    try{
      registration->Update();
    }
    catch( itk::ExceptionObject & err ){
      std::cerr << "ExceptionObject caught !" << std::endl;
      std::cerr << err << std::endl;
      return EXIT_FAILURE;
    }
 
    const double bestValue = optimizer->GetValue();
    std::cout << "Result = " << std::endl;
    std::cout << " Metric value  = " << bestValue          << std::endl;
 
    //Add the transfrom to the list of transfroms
    transforms.push_back( registration->GetTransform() );

    //Build composite transfrom
    typedef itk::CompositeTransform< double, 2 > CompositeTransform;
    CompositeTransform::Pointer composite = CompositeTransform::New();
    for( std::list< RegistrationType::TransformPointer >::iterator it = transforms.begin();
		    it != transforms.end(); ++it){

      composite->AddTransform( *it );
    }

    
    //Move the image into the frame of slice 0 
    typedef itk::ResampleImageFilter< ImageType2D, ImageType2D >    ResampleFilterType;
 
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

  std::stringstream outfile;
  outfile << prefix << "-affine-slice.nrrd";
  ImageIO< ImageType3D >::saveImage(alignedVolume, outfile.str() );




 
  return EXIT_SUCCESS;
}
