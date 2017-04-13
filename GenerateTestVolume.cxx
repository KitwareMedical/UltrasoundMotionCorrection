#include "itkImage.h"
#include "itkRescaleIntensityImageFilter.h"
#include <itkSmoothingRecursiveGaussianImageFilter.h>
#include "itkBinaryThresholdImageFilter.h"

#include "ImageIO.h"
#include <cmath>

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
#include <itkSmoothingRecursiveGaussianImageFilter.h>
  //Command line parsing
  TCLAP::CmdLine cmd("Generate test volume for breathing correction", ' ', "1");

  TCLAP::ValueArg<std::string> volumeArg("v","volume","volume image name", true, "",
      "filename");
  cmd.add(volumeArg);

  
  
  try{
    cmd.parse( argc, argv );
  } 
  catch (TCLAP::ArgException &e){ 
    std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; 
    return -1;
  }

 
  ImageType3D::RegionType region;
  ImageType3D::IndexType start;
  start[0] = 0;
  start[1] = 0;
  start[2] = 0;
 
  ImageType3D::SizeType size;
  size[0] = 50;
  size[1] = 50;
  size[2] = 50;
 
  region.SetSize(size);
  region.SetIndex(start);
 
  ImageType3D::Pointer image = ImageType3D::New();
  image->SetRegions(region);
  image->Allocate();
  
    
  //linear trend
  int x1 = 10;
  int y1 = 10;

  //constant
  int x2 = 40;
  int y2 = 10;

  //slow sinusoid
  int x3 = 15;
  int y3 = 35;
  for(int i=0; i< size[2]; i++){
    double shift = sin( i/(size[2]-1.0) * 8 * M_PI ) * 5;
    
    ImageType3D::IndexType index;
    index[2 ] = i;

    index[0] = (int) lround( x1 + (30.0 *i) / size[2]  + shift);
    index[1] = (int) lround(y1 + (30.0 *i) / size[2]);
    image->SetPixel(index, 1.f);
   
    index[0] = (int) lround(x2 + shift);
    index[1] = y2;
    image->SetPixel(index, 1.f);
    
    index[0] = (int) lround( x3 + sin( i/(size[2]-1.0) * 1.5 * M_PI ) * 10 + shift );
    index[1] = (int) lround(y3 + sin( i/(size[2]-1.0) * 1.5 * M_PI ) * 10);
    //image->SetPixel(index, 1.f);
  } 

    
  ImageType3D::SpacingType volumeSpacing = image->GetSpacing();
  GaussianFilterPointer smooth = GaussianFilter::New();
  GaussianFilter::SigmaArrayType sig = GaussianFilter::SigmaArrayType::Filled(0);
  sig[0] = 1.5 * volumeSpacing[0]; 
  sig[1] = 1.5 * volumeSpacing[1]; 
  sig[2] = 0.1 * volumeSpacing[2]; 
  smooth->SetSigmaArray(sig);
  smooth->SetInput( image );
  smooth->Update();
  image = smooth->GetOutput();

  typedef itk::BinaryThresholdImageFilter <ImageType3D, ImageType3D>
    BinaryThresholdImageFilterType;
  BinaryThresholdImageFilterType::Pointer thresholdFilter
    = BinaryThresholdImageFilterType::New();
  thresholdFilter->SetInput( image );
  thresholdFilter->SetLowerThreshold(0.005);
  thresholdFilter->SetUpperThreshold(100);
  thresholdFilter->SetInsideValue(1);
  thresholdFilter->SetOutsideValue(0);

  image = thresholdFilter->GetOutput();

  ImageIO<ImageType3D>::saveImage( image, volumeArg.getValue() );
  
  return EXIT_SUCCESS;
}
