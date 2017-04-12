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
#include "itkCompositeTransform.h"
#include "itkRescaleIntensityImageFilter.h"
#include <itkSmoothingRecursiveGaussianImageFilter.h>
#if ITK_VERSION_MAJOR < 4
#include "itkBSplineDeformableTransform.h"
#else
#include "itkBSplineTransform.h"
#endif

#include "ImageIO.h"


#include <tclap/CmdLine.h>

typedef  float  PixelType;
 
typedef itk::Image< PixelType, 2 >  ImageType2D;
typedef itk::Image< PixelType, 3 >  ImageType3D;
     
typedef typename itk::SmoothingRecursiveGaussianImageFilter<ImageType3D, ImageType3D> GaussianFilter;
typedef typename GaussianFilter::Pointer GaussianFilterPointer;

typedef typename itk::RescaleIntensityImageFilter<ImageType3D, ImageType3D> RescaleFilter;
typedef typename RescaleFilter::Pointer RescaleFilterPointer;

int main(int argc, char **argv )
{

  //Command line parsing
  TCLAP::CmdLine cmd("BSpline registration of 3D ultrasound bubble image", ' ', "1");

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

  TCLAP::ValueArg< int > gridXArg("","gridX","Number of grid points for Bspline", false, 5,
      "integer");
  cmd.add( gridXArg );

  TCLAP::ValueArg< int > gridYArg("","gridY","Number of grid points for Bspline", false, 5,
      "integer");
  cmd.add( gridYArg );

  TCLAP::ValueArg< int > gridX2Arg("","gridX2","Number of grid points for Bspline", false, 20,
      "integer");
  cmd.add( gridX2Arg );

  TCLAP::ValueArg< int > gridY2Arg("","gridY2","Number of grid points for Bspline", false, 20,
      "integer");
  cmd.add( gridY2Arg );



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
 
  int dim = dimArg.getValue();
  
  double s1 = smooth1Arg.getValue();
  double s2 = smooth2Arg.getValue();
  double s3 = smooth3Arg.getValue();

  // Get the  volume
  ImageType3D::Pointer volume = ImageIO<ImageType3D>::readImage( volumeArg.getValue() );      
  RescaleFilterPointer rescale = RescaleFilter::New();
  rescale->SetInput(volume);
  rescale->SetOutputMaximum(1.0);
  rescale->SetOutputMinimum(0.0);
  rescale->Update();
      
  ImageType3D::SpacingType volumeSpacing = volume->GetSpacing();
  GaussianFilterPointer smooth = GaussianFilter::New();
  GaussianFilter::SigmaArrayType sig = GaussianFilter::SigmaArrayType::Filled(0);
  sig[0] = s1 * volumeSpacing[0]; 
  sig[1] = s2 * volumeSpacing[1]; 
  sig[2] = s3 * volumeSpacing[2]; 
  smooth->SetSigmaArray(sig);
  smooth->SetInput( rescale->GetOutput() );
  smooth->Update();
  volume = smooth->GetOutput();
  
  {
    std::stringstream outfile;
    outfile << prefix << "-smoothed-" << s1 << "-" << s2 << "-" << s3 << ".nrrd";
    ImageIO<ImageType3D>::saveImage( volume, outfile.str()  );
  }

  ImageType3D::RegionType volumeRegion = volume->GetLargestPossibleRegion();
  ImageType3D::SizeType volumeSize = volumeRegion.GetSize();

  //List for strong trsanfroms between slices
  typedef itk::ImageRegistrationMethod< ImageType2D, ImageType2D >    RegistrationType;
  
  //Setup bspline transfrom registratiom  
  const unsigned int SpaceDimension = 2;
  const unsigned int SplineOrder = 2;
  typedef double CoordinateRepType;
  typedef itk::BSplineTransform< CoordinateRepType,SpaceDimension,  SplineOrder >     TransformType;

  std::list< TransformType::Pointer > transforms;
  
  //Store moved images
  //std::vector< ImageType2D::Pointer > aligned( volumeSize[0] );

  //Between each slice in the volume compute an bspline transfrom
  for(int i=1; i < volumeSize[dim]; i++){


    //Extract consecutive slices
    ImageType3D::IndexType fixedStart;
    fixedStart.Fill(0);
    fixedStart[dim] = i-1;
    
    ImageType3D::IndexType movingStart;
    movingStart.Fill(0);
    movingStart[dim] = i;
 
    ImageType3D::SizeType size = volumeSize;
    size[dim] = 0;

    std::cout << fixedStart << std::endl;
    std::cout << size << std::endl; 
    ImageType3D::RegionType fixedRegion(fixedStart, size);
    ImageType3D::RegionType movingRegion(movingStart, size);
 
 
    typedef itk::ExtractImageFilter< ImageType3D, ImageType2D > ExtractFilter;
    ExtractFilter::Pointer fixedExtract = ExtractFilter::New();
    fixedExtract->SetExtractionRegion(fixedRegion);
    fixedExtract->SetInput(volume);
    fixedExtract->SetDirectionCollapseToIdentity(); // This is required.
    fixedExtract->Update();
 
    ImageType2D::Pointer fixedImage = fixedExtract->GetOutput();

    ExtractFilter::Pointer movingExtract = ExtractFilter::New();
    movingExtract->SetExtractionRegion(movingRegion);
    movingExtract->SetInput(volume);
    movingExtract->SetDirectionCollapseToIdentity(); // This is required.
    movingExtract->Update();
 
    ImageType2D::Pointer movingImage = movingExtract->GetOutput();


    if(i==1){
      //aligned[0] = fixedImage;
    }

    TransformType::Pointer      transform1     = TransformType::New();
    TransformType::PhysicalDimensionsType   fixedPhysicalDimensions1;
    TransformType::MeshSizeType             meshSize1;
    for( unsigned int j=0; j < 2; j++ ){
	    fixedPhysicalDimensions1[j] = fixedImage->GetSpacing()[j] *
		    static_cast<double>(fixedImage->GetLargestPossibleRegion().GetSize()[j] - 1 );
    }
    meshSize1[0] = gridXArg.getValue();
    meshSize1[1] = gridYArg.getValue();
    transform1->SetTransformDomainOrigin( fixedImage->GetOrigin() );
    transform1->SetTransformDomainPhysicalDimensions( fixedPhysicalDimensions1 );
    transform1->SetTransformDomainMeshSize( meshSize1 );
    transform1->SetTransformDomainDirection( fixedImage->GetDirection() );

    typedef TransformType::ParametersType     ParametersType;
    const unsigned int numberOfParameters1 = transform1->GetNumberOfParameters();
    ParametersType parameters1( numberOfParameters1 );
    parameters1.Fill( 0.0 );
    transform1->SetParameters( parameters1 );


    TransformType::Pointer      transform2     = TransformType::New();
    TransformType::PhysicalDimensionsType   fixedPhysicalDimensions2;
    TransformType::MeshSizeType             meshSize2;
    for( unsigned int j=0; j < 2; j++ ){
	    fixedPhysicalDimensions2[j] = fixedImage->GetSpacing()[j] *
		    static_cast<double>(fixedImage->GetLargestPossibleRegion().GetSize()[j] - 1 );
    }
    meshSize2[0] = gridX2Arg.getValue();
    meshSize2[1] = gridY2Arg.getValue();
    transform2->SetTransformDomainOrigin( fixedImage->GetOrigin() );
    transform2->SetTransformDomainPhysicalDimensions( fixedPhysicalDimensions2 );
    transform2->SetTransformDomainMeshSize( meshSize2 );
    transform2->SetTransformDomainDirection( fixedImage->GetDirection() );
    
    const unsigned int numberOfParameters2 = transform2->GetNumberOfParameters();
    ParametersType parameters2( numberOfParameters2 );
    parameters2.Fill( 0.0 );
    transform2->SetParameters( parameters2 );

    typedef itk::CompositeTransform< double, 2 > CompositeTransform;
    CompositeTransform::Pointer transform = CompositeTransform::New();
    transform->AddTransform(transform2);
    transform->AddTransform(transform1);
    transform->SetAllTransformsToOptimizeOn();

    typedef CompositeTransform::ParametersType     CParametersType;
 
    const unsigned int numberOfParameters =
               transform->GetNumberOfParameters();


    std::cout << numberOfParameters << std::endl; 
    CParametersType parameters( numberOfParameters );
    parameters.Fill( 0.0 );
    transform->SetParameters( parameters );


    //typedef itk::GradientDescentOptimizer       OptimizerType;
    typedef itk::LBFGSOptimizer       OptimizerType;
    typedef itk::MeanSquaresImageToImageMetric< ImageType2D, ImageType2D >  MetricType;
    typedef itk::LinearInterpolateImageFunction< ImageType2D, double >    InterpolatorType;
 
        
    MetricType::Pointer         metric        = MetricType::New();
    OptimizerType::Pointer      optimizer     = OptimizerType::New();
    InterpolatorType::Pointer   interpolator  = InterpolatorType::New();
    RegistrationType::Pointer   registration  = RegistrationType::New();


    registration->SetMetric(        metric        );
    registration->SetOptimizer(     optimizer     );
    registration->SetTransform(     transform     );
    registration->SetInterpolator(  interpolator  );
    registration->SetFixedImage(    fixedImage    );
    registration->SetMovingImage(   movingImage   );
    registration->SetFixedImageRegion( fixedImage->GetLargestPossibleRegion() );
    registration->SetInitialTransformParameters( transform->GetParameters() );
 
    //optimizer->SetMaximumStepLength( .1 ); 
    //optimizer->SetMinimumStepLength( 0.0001 );
    
     //optimizer->SetLearningRate( 0.5 );
     //optimizer->SetNumberOfIterations( 200 );
      
    optimizer->SetGradientConvergenceTolerance( 0.00000001 );
    optimizer->SetLineSearchAccuracy( 0.9 );
    optimizer->SetDefaultStepLength( 0.1 );
    optimizer->TraceOn();
    optimizer->SetMaximumNumberOfFunctionEvaluations( 200 );

    try{
      registration->SetNumberOfThreads(1);
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
    //CompositeTransform::Pointer tmp = dynamic_cast< CompositeTransform >( registration->GetTransform() );
    transforms.push_back( transform1 );

    //Build composite transfrom
    CompositeTransform::Pointer composite = CompositeTransform::New();
    for( std::list< TransformType::Pointer >::iterator it = transforms.begin();
		    it != transforms.end(); ++it){

      composite->AddTransform( *it );
    }

    //Move the image into the frame of slice 0 
    typedef itk::ResampleImageFilter< ImageType2D, ImageType2D >    ResampleFilterType;
 
    ResampleFilterType::Pointer resampler = ResampleFilterType::New();
    resampler->SetInput( movingImage );
 
    resampler->SetTransform( composite );
 
    resampler->SetSize( fixedImage->GetLargestPossibleRegion().GetSize() );
    resampler->SetOutputOrigin(  fixedImage->GetOrigin() );
    resampler->SetOutputSpacing( fixedImage->GetSpacing() );
    resampler->SetOutputDirection( fixedImage->GetDirection() );
    resampler->SetDefaultPixelValue( 0 );
    resampler->Update();
    ImageType2D::Pointer moved = resampler->GetOutput();

    //aligned[i] = moved;
	 
    std::stringstream outfile;
    outfile << prefix << "-bspline-" << i << ".nrrd";
    ImageIO<ImageType2D>::saveImage( moved, outfile.str()  );

    if( i== 1){
      std::stringstream outfile;
      outfile << prefix << "-bspline-" << 0 << ".nrrd";
      ImageIO<ImageType2D>::saveImage( fixedImage, outfile.str() );
    }
  }

  return EXIT_SUCCESS;
}
