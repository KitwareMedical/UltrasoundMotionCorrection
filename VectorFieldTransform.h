#ifndef VECTORFIELDTRANSFORMATION_H
#define VECTORFIELDTRANSFORMATION_H


#include "itkImageRegionIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "VectorImage.h"

#include "ImageIO.h"

template<typename TImage, typename TPrecision, typename InterpolateFunction>
class VectorFieldTransform{
  
  public:
    typedef TImage Image;
    typedef typename Image::Pointer ImagePointer;
    typedef typename Image::PixelType ImagePixel;
    typedef typename Image::IndexType ImageIndex;
    typedef typename Image::RegionType ImageRegion;
    typedef typename Image::SizeType ImageSize;
    
    typedef typename InterpolateFunction::OutputType InterpolateOutput;
    
    typedef VectorImage<TImage, Image::ImageDimension> VImage;
    typedef typename VImage::VectorType VectorType;
    typedef typename VImage::CoVectorType CoVectorType;

    typedef typename VImage::ITKVectorImage ITKVImage;
    typedef typename ITKVImage::Pointer ITKVImagePointer;

    static VImage  *InitializeZeroTransform(ImagePointer image){
      VImage *transform = new VImage();
      transform->createZero(image);
      return transform;
    };
    
    
    static ImagePointer Transform(ImagePointer image,
        ITKVImagePointer transform){
     
      ImagePointer transformed = ImageIO<Image>::createImage(image);
      Transform(transformed, image, transform);
      return transformed;

    };

    
    // transformed(x) = image(x + transform(x));
    static void Transform(ImagePointer transformed, ImagePointer image,
        ITKVImagePointer transform){
     
      InterpolateFunctionPointer imageip = InterpolateFunction::New();
      imageip->SetInputImage( image );

      ImageRegion region = transformed->GetLargestPossibleRegion();
      ImageSize size = region.GetSize();
      ImageIndex index = region.GetIndex();


      ImageIterator it( transformed, region);
      ITKVImageIterator vit(transform, region);

      CoVectorType v(image->GetImageDimension() ); 
      for(it.GoToBegin(); !it.IsAtEnd(); ++it, ++vit){
        ImageIndex current = it.GetIndex();
        v = vit.Get();
	Point p;
        transformed->TransformIndexToPhysicalPoint(current, p); 
        for(int i=0; i<v.GetNumberOfComponents(); i++){
	  p[i] += v[i];
	}

        ImageContinuousIndex cindex;
        transformed->TransformPhysicalPointToContinuousIndex(p, cindex);
        for(unsigned int i=0; i<ImageIndex::GetIndexDimension(); i++){
          //check range of index
          if(cindex[i] < index[i]){
            cindex[i] = index[i];
          }
          else if(cindex[i] > index[i] + size[i] - 1){
            cindex[i] = index[i] + size[i] - 1;
          }
        }


        //Get pixel from source image
        InterpolateOutput ipval = imageip->EvaluateAtContinuousIndex( cindex );
        it.Set( ipval );

      
      }

    };


    // transformed(x) = image(x + transform(x));
    static void Transform(ImagePointer transformed, ImagePointer image,
        VImage *transform){
     
      if(transform == NULL){
        return;
      } 

      InterpolateFunctionPointer imageip = InterpolateFunction::New();
      imageip->SetInputImage( image );

      ImageRegion region = transformed->GetLargestPossibleRegion();
      ImageSize size = region.GetSize();
      ImageIndex index = region.GetIndex();


      ImageIterator it( transformed, region);
      transform->initIteration(region);
      transform->goToBegin();
      VectorType v(image->GetImageDimension()); 
      for(it.GoToBegin(); !it.IsAtEnd(); ++it, transform->next()){
        ImageIndex current = it.GetIndex();
        transform->get(v);
        
        Point p;
        transformed->TransformIndexToPhysicalPoint(current, p);
        p += v;
       
        ImageContinuousIndex cindex;
        transformed->TransformPhysicalPointToContinuousIndex(p, cindex);
        for(unsigned int i=0; i<ImageIndex::GetIndexDimension(); i++){
          //check range of index
          if(cindex[i] < index[i]){
            cindex[i] = index[i];
          }
          else if(cindex[i] > index[i] + size[i] - 1){
            cindex[i] = index[i] + size[i] - 1;
          }
        }


        //Get pixel from source image
        InterpolateOutput ipval = imageip->EvaluateAtContinuousIndex( cindex );
        it.Set( ipval );
      
      }

    };

// transformed(x) = image(x + transform(x));
    static ImagePointer Transform(ImagePointer image,
        VImage *transform){
     
      ImagePointer transformed = ImageIO<Image>::copyImage(image);
      Transform(transformed, image, transform);
      return transformed;

    };

  private:

    VectorFieldTransform(){};



    typedef typename itk::ImageRegionIteratorWithIndex<Image> ImageIterator;
    typedef typename itk::ImageRegionIteratorWithIndex<ITKVImage> ITKVImageIterator;

    typedef typename InterpolateFunction::Pointer InterpolateFunctionPointer;
    typedef typename InterpolateFunction::ContinuousIndexType ImageContinuousIndex;
    
    typedef typename itk::Point<TPrecision, Image::ImageDimension > Point;
};


#endif
