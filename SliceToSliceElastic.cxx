#include "itkExtractImageFilter.h"
#include "itkImage.h"
#include "itkImageRegistrationMethod.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkGradientDescentOptimizer.h"
#include "itkResampleImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkSpatialObjectToImageFilter.h"
#include "itkAffineTransform.h"
#include "itkCompositeTransform.h"
#include "itkRescaleIntensityImageFilter.h"
#include <itkSmoothingRecursiveGaussianImageFilter.h>
#include "ImageIO.h"

#include "ElasticWarp.h"

#include <tclap/CmdLine.h>

typedef  float  PixelType;
 
typedef itk::Image< PixelType, 2 >  ImageType2D;
typedef itk::Image< PixelType, 3 >  ImageType3D;
     
typedef typename itk::SmoothingRecursiveGaussianImageFilter<ImageType3D, ImageType3D> GaussianFilter;
typedef typename GaussianFilter::Pointer GaussianFilterPointer;

typedef typename itk::RescaleIntensityImageFilter<ImageType3D, ImageType3D> RescaleFilter;
typedef typename RescaleFilter::Pointer RescaleFilterPointer;

typedef SimpleWarp<ImageType2D> Warp; 

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

  
  TCLAP::ValueArg<double> stepArg("s","scaling",
      "scaling of maximum step size for gradient descent", false, (double)0.2, 
      "step size");
  cmd.add(stepArg);

  TCLAP::ValueArg<int> iterArg("i","iterations",
      "maximum number of iterations per scale", false, 200, 
      "int");
  cmd.add(iterArg);

  TCLAP::ValueArg<double> alphaArg("","alpha",
      "weight of gradient (alpha) and identity (1-alpha) penalty", false, (double) 0.5, 
      "float");
  cmd.add(alphaArg);
  
  TCLAP::ValueArg<double> lambdaArg("","lambda",
      "inital weight of image differnce term", false, (double) 1, 
      "float");
  cmd.add(lambdaArg);

  TCLAP::ValueArg<double> lambdaIncArg("","lambdaInc",
      "increase in lambda when images can not be registered within epsilon for inital given lambda ", 
      false, (double) 0.1, 
      "float");
  cmd.add(lambdaIncArg);
  
  TCLAP::ValueArg<double> lambdaIncTArg("","lambdaIncThreshold",
      "Threshold in intnesity difference between two consectuive updates to update lambda", 
      false, (double) 0.0001, 
      "float");
  cmd.add(lambdaIncTArg);

  TCLAP::ValueArg<double> epsArg("","epsilon",
      "Maximal root mean squared intensity difference to register images to", false, (double) 0.01, 
      "tolerance");
  cmd.add(epsArg);

  TCLAP::ValueArg<double> sigmaArg("","sigma",
      "Multiscale sigma of coarsest scale"
      , false, (double) 2, "float");
  cmd.add(sigmaArg);



  TCLAP::ValueArg<int> nscaleArg("","nscales",
      "Number of scales. 0 corresponds to no scales", false, 5, 
      "int");
  cmd.add(nscaleArg);


  TCLAP::ValueArg<std::string> prefixArg("p","prefix","Prefix for stroing output images", true, "",
      "filename");
  cmd.add(prefixArg);
 
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
  std::list< Warp::VImage *> transforms;
  
  //Vector for storing moved images
  std::vector< ImageType2D::Pointer > aligned( volumeSize[0] );

  int dim = dimArg.getValue();
  //Between each slice in the volume compute an affine transfrom
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
#if ITK_VERSION_MAJOR >= 4
    fixedExtract->SetDirectionCollapseToIdentity(); // This is required.
#endif
    fixedExtract->Update();
 
    ImageType2D::Pointer fixedImage = fixedExtract->GetOutput();

    ExtractFilter::Pointer movingExtract = ExtractFilter::New();
    movingExtract->SetExtractionRegion(movingRegion);
    movingExtract->SetInput(volume);
#if ITK_VERSION_MAJOR >= 4
    movingExtract->SetDirectionCollapseToIdentity(); // This is required.
#endif
    movingExtract->Update();
 
    ImageType2D::Pointer movingImage = movingExtract->GetOutput();


    if(i==1){
      aligned[0] = fixedImage;
    }

    //Setup elastic registration
    //Setup warp
    ImageType2D::Pointer mask = ImageIO<ImageType2D>::copyImage(fixedImage);
    //set mask to all ones = no masking
    itk::ImageRegionIterator<ImageType2D> it(mask, mask->GetLargestPossibleRegion());
    for(; !it.IsAtEnd(); ++it){
     it.Set(1);
    }

    Warp warp;
    warp.setAlpha(alphaArg.getValue());
    warp.setMaximumIterations(iterArg.getValue());
    warp.setMaximumMotion(stepArg.getValue());
    warp.setEpsilon(epsArg.getValue());
    warp.setLambda(lambdaArg.getValue());
    warp.setLambdaIncrease(lambdaIncArg.getValue());
    warp.setLambdaIncreaseThreshold(lambdaIncTArg.getValue());
    
    Warp::VImage *vectorfield = warp.warpMultiresolution( movingImage, fixedImage, mask,
                                                    nscaleArg.getValue(), sigmaArg.getValue() );
 
    //Align vectorfeilds
    ImageType2D::Pointer *vc = vectorfield->getComps(); 
        
    {
      std::stringstream outfile;
      outfile << prefix << "-elastic-vx-" << i << ".nrrd";
      ImageIO<ImageType2D>::saveImage( vc[0], outfile.str()  );
    }
    {
      std::stringstream outfile;
      outfile << prefix << "-elastic-vy-" << i << ".nrrd";
      ImageIO<ImageType2D>::saveImage( vc[1], outfile.str()  );
    }
    if( i==1 ){
      Warp::VImage zero;
      zero.createZero( vc[0] );
      {
      std::stringstream outfile;
      outfile << prefix << "-elastic-vx-" << 0 << ".nrrd";
      ImageIO<ImageType2D>::saveImage( zero.getComps()[0], outfile.str()  );
      }
      {
      std::stringstream outfile;
      outfile << prefix << "-elastic-vy-" << 0 << ".nrrd";
      ImageIO<ImageType2D>::saveImage( zero.getComps()[1], outfile.str()  );
      }
    }

    for( std::list< Warp::VImage * >::reverse_iterator it = transforms.rbegin();
		    it != transforms.rend(); ++it){

       vc[0] = Warp::ImageTransform::Transform(vc[0], *it);
       vc[1] = Warp::ImageTransform::Transform(vc[1], *it);
    }

    {
      std::stringstream outfile;
      outfile << prefix << "-elastic-avx-" << i << ".nrrd";
      ImageIO<ImageType2D>::saveImage( vc[0], outfile.str()  );
    }
    {
      std::stringstream outfile;
      outfile << prefix << "-elastic-avy-" << i << ".nrrd";
      ImageIO<ImageType2D>::saveImage( vc[1], outfile.str()  );
    }


    //Add the transfrom to the list of transfroms
    transforms.push_back( vectorfield );

    ImageType2D::Pointer movedOS = Warp::ImageTransform::Transform(movingImage, vectorfield); 
    //Build composite transfrom
    ImageType2D::Pointer moved = movingImage; 
    for( std::list< Warp::VImage * >::reverse_iterator it = transforms.rbegin();
		    it != transforms.rend(); ++it){

       moved = Warp::ImageTransform::Transform(moved, *it);
    }

    { 
    std::stringstream outfile;
    outfile << prefix << "-elastic-" << i << ".nrrd";
    ImageIO<ImageType2D>::saveImage( moved, outfile.str()  );
    }
    {
      std::stringstream outfile;
      outfile << prefix << "-elastic-os-" << i << ".nrrd";
      ImageIO<ImageType2D>::saveImage( movedOS, outfile.str()  );
    }
    if( i== 1){
      {
      std::stringstream outfile;
      outfile << prefix << "-elastic-" << 0 << ".nrrd";
      ImageIO<ImageType2D>::saveImage( fixedImage, outfile.str() );
      }
      {
      std::stringstream outfile;
      outfile << prefix << "-elastic-os-" << 0 << ".nrrd";
      ImageIO<ImageType2D>::saveImage( fixedImage, outfile.str() );
      }
    }

  }

 
  return EXIT_SUCCESS;
}
