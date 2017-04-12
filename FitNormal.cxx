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
#include <itkNeighborhoodIterator.h>
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
  TCLAP::CmdLine cmd("Affine registration of 3D ultrasound bubble image", ' ', "1");

  TCLAP::ValueArg<std::string> volumeArg("v","volume","Ultrasound volume image", true, "",
      "filename");
  cmd.add(volumeArg);

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
  RescaleFilterPointer rescale = RescaleFilter::New();
  rescale->SetInput(volume);
  rescale->SetOutputMaximum(1.0);
  rescale->SetOutputMinimum(0.0);
  rescale->Update();

  volume = rescale->GetOutput();

      
  ImageType3D::SpacingType volumeSpacing = volume->GetSpacing();
  ImageType3D::RegionType volumeRegion = volume->GetLargestPossibleRegion();
  ImageType3D::SizeType volumeSize = volumeRegion.GetSize();

 
  VImage transform;
  transform.createZero( volume );
  typedef typename itk::NeighborhoodIterator<ImageType3D> NIterator;
  NIterator::SizeType radius;
  int r1 = std::ceil( s1 * 3 );
  int r2 = std::ceil( s2 * 3 );
  int r3 = std::ceil( s3 * 3 );
  radius[0] = r1;
  radius[1] = r2;
  radius[2] = r3;
  NIterator it( radius, volume, volume->GetLargestPossibleRegion() );
  //it.SetToBegin();

  transform.initIteration( volume->GetLargestPossibleRegion() );
  transform.goToBegin();

  itk::NeighborhoodIterator<ImageType3D>::OffsetType offsset; 
  //int count = 0;
  double s12 = s1*s1;
  double s22 = s2*s2;
  double s32 = s3*s3;
  while( !it.IsAtEnd() ){
    //Estimate displacement
    //current location - weighted mean of neighboring pixel locations
    VImage::VectorType v;
    v.Fill(0);
    NIterator::OffsetType offset;
    offset.Fill(0);
    double sumw = 0;
    for(int i=-r1; i<=r1; i++){
      for(int j=-r2; j<=r2; j++){
        for(int k=-r3; k<=r3; k++){
	  offset[0] = i;
	  offset[1] = j;
	  offset[2] = k;
	  double w = it.GetPixel( offset );
	  double a = exp(-(i*i) / s12  - (j*j) / s22   - (k*k) / s32 );
	  //std::cout << w << " / " << a << std::endl;
	  w *= a;
	  sumw += w;  	  
	  v[0] -= w * i * volumeSpacing[0];
	  v[1] -= w * j * volumeSpacing[0];
	  v[2] -= w * k * volumeSpacing[0];

	}
      }
    }
    if(sumw > 0){
      v /= sumw;
    }
    transform.set( v );
    ++it;
    transform.next();
    //count++;
    //std::cout << count <<  " ";
  }

  ImageType3D::Pointer moved = VTransform::Transform( volume, &transform);
  std::stringstream outfile;
  outfile << prefix << "-fit-moved.nrrd";
  ImageIO< ImageType3D >::saveImage( moved, outfile.str()  );
  {
  std::stringstream outfile;
  outfile << prefix << "-fit-transform.nrrd";
  ImageIO< ITKVImage >::saveImage( transform.toITK(), outfile.str()  );
  }

  return EXIT_SUCCESS;
}
