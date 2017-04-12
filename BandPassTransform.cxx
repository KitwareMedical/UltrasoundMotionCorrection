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
#include <itkLinearInterpolateImageFunction.h>

#include "ImageIO.h"

#include "VectorImage.h"
#include "VectorFieldTransform.h"

#include <tclap/CmdLine.h>

typedef  float  PixelType;
 
typedef itk::Image< PixelType, 2 >  ImageType2D;
typedef itk::Image< PixelType, 3 >  ImageType3D;
 
typedef VectorImage<ImageType3D, 3> VImage;
typedef typename VImage::ITKVectorImage ITKVImage;
typedef typename itk::LinearInterpolateImageFunction<ImageType3D, double> LinearInterpolate;
typedef VectorFieldTransform<ImageType3D, double, LinearInterpolate> VTransform;    

typedef typename itk::SmoothingRecursiveGaussianImageFilter<ImageType3D, ImageType3D> GaussianFilter;
typedef typename GaussianFilter::Pointer GaussianFilterPointer;

typedef typename itk::RescaleIntensityImageFilter<ImageType3D, ImageType3D> RescaleFilter;
typedef typename RescaleFilter::Pointer RescaleFilterPointer;


int main(int argc, char **argv )
{

  //Command line parsing
  TCLAP::CmdLine cmd("Band pass filter a vector transform", ' ', "1");

  TCLAP::ValueArg<std::string> volumeArg("v","volume","Ultrasound volume image", true, "",
      "filename");
  cmd.add(volumeArg);
   
  TCLAP::ValueArg<std::string> transformXArg("","tx","X-component of vectorfield for transfromation", true, "",
      "filename");
  cmd.add(transformXArg);   
  
  TCLAP::ValueArg<std::string> transformYArg("","ty","Y-component of vectorfield for transfromation", true, "",
      "filename");
  cmd.add(transformYArg);   

  TCLAP::ValueArg<std::string> prefixArg("p","prefix","Prefix for stroing output images", true, "",
      "filename");
  cmd.add(prefixArg);
 
  TCLAP::ValueArg<float> low1Arg("","low1", "Low pass filter sigma x", false, 1.0, "float");
  cmd.add(low1Arg);
  TCLAP::ValueArg<float> low2Arg("","low2", "Low pass filter sigma y", false, 1.0, "float");
  cmd.add(low2Arg);
  TCLAP::ValueArg<float> low3Arg("","low3", "Low pass filter sigma z", false, 1.0, "float");
  cmd.add(low3Arg);
  
  TCLAP::ValueArg<float> high1Arg("","high1", "Low pass filter sigma x", false, 1.0, "float");
  cmd.add(high1Arg);
  TCLAP::ValueArg<float> high2Arg("","high2", "Low pass filter sigma y", false, 1.0, "float");
  cmd.add(high2Arg);
  TCLAP::ValueArg<float> high3Arg("","high3", "Low pass filter sigma z", false, 1.0, "float");
  cmd.add(high3Arg);

  
  try{
    cmd.parse( argc, argv );
  } 
  catch (TCLAP::ArgException &e){ 
    std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; 
    return -1;
  }

  std::string prefix = prefixArg.getValue();
  
  double l1 = low1Arg.getValue();
  double l2 = low2Arg.getValue();
  double l3 = low3Arg.getValue();
  double h1 = high1Arg.getValue();
  double h2 = high2Arg.getValue();
  double h3 = high3Arg.getValue();


  // Get the  volume
  ImageType3D::Pointer volume = ImageIO<ImageType3D>::readImage( volumeArg.getValue() );    
  ImageType3D::Pointer vx = ImageIO<ImageType3D>::readImage( transformXArg.getValue() );    
  ImageType3D::Pointer vy = ImageIO<ImageType3D>::readImage( transformYArg.getValue() );    
  //ITKVImage::Pointer vim = ImageIO< ITKVImage >::readImage( transformArg.getValue() );

  VImage v1;
  v1.createZero(vx);
  v1.setComp(0, vx);
  v1.setComp(1, vy);
  
  VImage vh;
  vh.copy(&v1);
  

  VImage::GaussianFilter::SigmaArrayType low;
  low[0] = low1Arg.getValue();
  low[1] = low2Arg.getValue();
  low[2] = low3Arg.getValue();
  VImage::GaussianFilter::SigmaArrayType high;
  high[0] = high1Arg.getValue();
  high[1] = high2Arg.getValue();
  high[2] = high3Arg.getValue();
  
  vh.blur( high );
  {
  std::stringstream outfile;
  outfile << prefix << "-bandpass-transform-high.nrrd";
  ImageIO<ITKVImage>::saveImage( vh.toITK(), outfile.str() );
  }
  
  vh.multiply(-1);
  v1.add(&vh);
  v1.blur(low);
 
  {
  std::stringstream outfile;
  outfile << prefix << "-bandpass-transform.nrrd";
  ImageIO<ITKVImage>::saveImage( v1.toITK(), outfile.str() );
  }

  ImageType3D::Pointer moved = VTransform::Transform( volume, &v1 ); 
  {
  std::stringstream outfile;
  outfile << prefix << "-bandpass-moved.nrrd";
  ImageIO<ImageType3D>::saveImage( moved, outfile.str() );
  }


  return EXIT_SUCCESS;
}
