#ifndef SIMPLEWARP_H
#define SIMPLEWARP_H

#include "VectorFieldTransform.h"


#include "itkGradientImageFilter.h"
#include "itkImageRegionIterator.h"
#include "itkMinimumMaximumImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkVectorLinearInterpolateImageFunction.h"
#include "itkBSplineDownsampleImageFilter.h"
#include "itkBSplineUpsampleImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkPoint.h"

#include <algorithm>

#define VERBOSE

template<typename TImage>
class SimpleWarp{

  public:


    typedef TImage Image;
    typedef typename Image::Pointer ImagePointer;
    typedef typename Image::PixelType TPrecision;
    typedef typename Image::IndexType ImageIndex;
    typedef typename Image::SpacingType ImageSpacing;
    typedef typename Image::RegionType ImageRegion;
    typedef typename Image::SizeType ImageSize;

    typedef typename itk::LinearInterpolateImageFunction<Image, TPrecision>
      LinearInterpolate;
    typedef typename LinearInterpolate::Pointer LinearInterpolatePointer;
    typedef typename LinearInterpolate::ContinuousIndexType ImageContinuousIndex;

    typedef VectorFieldTransform<Image, TPrecision, LinearInterpolate>
      ImageTransform;

    typedef typename ImageTransform::VImage VImage;
    typedef typename VImage::VectorType VectorType;
    typedef typename VImage::ITKVectorImagePointer ITKVectorImagePointer;
    typedef typename VImage::ITKVectorImage ITKVectorImage;

    SimpleWarp(){
      nIterations = 200;
      alpha = 0.5;
      lambda = 1;
      lambdaInc = 0.1;
      lambdaIncThreshold = 0.0001;
      maxMotion = 0.8;
      epsilon = 0.01;
    };



    //Do a multires warp starting with nres resolutions
    VImage *warpMultiresolution(ImagePointer i1, ImagePointer i2, ImagePointer
        mask, int nres, TPrecision sigma){
      
      ImagePointer *pyramid1 = downsample(i1, nres, sigma);
      ImagePointer *pyramid2 = downsample(i2, nres, sigma);
      ImagePointer *pyramidMask =  downsample(mask, nres, sigma);
      //ImageIO<Image>::saveImage(pyramid1[nres-1], "p1.mhd");
      //ImageIO<Image>::saveImage(pyramid2[nres-1], "p2.mhd");


      //transform images
      VImage *transform = NULL;
      VImage *prevtransform = NULL;

#ifdef VERBOSE
      std::cout << "Computing Multires pyramid" << std::endl;
#endif
      TPrecision lambdaTmp = lambda;
      TPrecision lambdaIncThresholdTmp = lambdaIncThreshold;
      //TPrecision maxMotionTmp = maxMotion;
      //Multires steps
      for(int i=nres-1; i>0; i--){
        //do warp
	      lambda = lambdaTmp * pow(2, i);
        lambdaIncThreshold = lambdaIncThresholdTmp / pow(2, i);
        //maxMotion = maxMotionTmp / pow(2, i);
        std::cout << lambda << " | " << lambdaIncThreshold << std::endl;
        transform = warp(pyramid1[i], pyramid2[i], pyramidMask[i], prevtransform);
        delete prevtransform;
        prevtransform = transform;

        //std::stringstream ss;
        //ss << "dscale_" << i << ".nrrd";
        //ITKVectorImagePointer i1 = prevtransform->toITK();
        //ImageIO<ITKVectorImage>::saveImage(i1, ss.str() );
        upsample(prevtransform, pyramid1[i-1]->GetLargestPossibleRegion().GetSize(), pyramid1[i-1]->GetSpacing());
        //std::stringstream ss1;
        //ss1 << "dscale_" << i << "u.nrrd";
        //ITKVectorImagePointer i2 = prevtransform->toITK();
        //ImageIO<ITKVectorImage>::saveImage( i2, ss1.str() );
      }

#ifdef VERBOSE
      std::cout << "Computing Multires pyramid - done" << std::endl;
#endif
      lambda = lambdaTmp;
      lambdaIncThreshold = lambdaIncThresholdTmp;
      //maxMotion = maxMotionTmp;
      transform = warp(pyramid1[0], pyramid2[0], pyramidMask[0], prevtransform);
      delete prevtransform;
      
      delete[] pyramid1;
      delete[] pyramid2;
      delete[] pyramidMask;

      return transform;    
    };



    //Do a multiscale warp starting with a gaussian blur with var = sigma^2
    VImage *warpMultiscale(ImagePointer i1, ImagePointer i2, ImagePointer mask,
        TPrecision sigma, int nsteps, VImage *initTransform = NULL){
      
      VImage *transform = NULL;
      VImage *prevTransform = initTransform;
      if(prevTransform != NULL){
        prevTransform = new VImage();
        prevTransform->copy(initTransform);
      }

      //Multiscale steps
      for(int i=0; i<nsteps; i++){
        GaussianFilterPointer smooth1 = GaussianFilter::New();
        smooth1->SetSigma(sigma);
        smooth1->SetInput(i1);
        smooth1->Update();
        GaussianFilterPointer smooth2 = GaussianFilter::New();
        smooth2->SetSigma(sigma);
        smooth2->SetInput(i2);
        smooth2->Update();

        transform = warp(smooth1->GetOutput(), smooth2->GetOutput(), mask,
            prevTransform);
        delete prevTransform;
        prevTransform = transform;

        sigma = sigma/2;
      }

      transform = warp(i1, i2, mask, prevTransform);
      delete prevTransform;
      return transform;    
    };


    //Compute vectorfield v minimizing  
    // Int ( i1(x + v(x)) - i2 ) dx + alpha * Int | grad v(x) |^2 dx  
    VImage *warp(ImagePointer i1, ImagePointer i2, ImagePointer mask, 
        VImage *initTransform = NULL){

      static TPrecision EPSILON = 1.0e-20;
      const int dimension = Image::ImageDimension;
            
      ImageSpacing spacing = i1->GetSpacing(); 
      //std::cout << spacing << std::endl;
      TPrecision scaling = spacing[0];
      for(unsigned int i=1; i< ImageIndex::GetIndexDimension(); i++){
        scaling = std::min((TPrecision)scaling, (TPrecision)spacing[i]);
      }

      //Transformation vector field (v)
      VImage *transform;
      if(initTransform == NULL){
        transform = ImageTransform::InitializeZeroTransform(i1);
      }
      else{
        transform = new VImage();
        transform->copy(initTransform);
      }
      VImage *transformUpdate = ImageTransform::InitializeZeroTransform(i1); 

      //Interpolate function for obtaining i1(x+v(x))
      LinearInterpolatePointer i1Interpolate = LinearInterpolate::New();
      i1Interpolate->SetInputImage(i1);

      LinearInterpolatePointer i2Interpolate = LinearInterpolate::New();
      i2Interpolate->SetInputImage(i2);
      
      //Gradient of target image
      GradientImageFilterPointer grad = GradientImageFilter::New();
      grad->SetUseImageSpacing(true);
      grad->SetUseImageDirection(true);
      grad->SetInput(i1);
      grad->Update(); 
      GradientImagePointer gi1 = grad->GetOutput();
      itk::Vector<TPrecision, dimension> tmpVector;

      //Interpolate function for obtaining  grad(i1)(x+v(x))
      GradientLinearInterpolatePointer gi1Interpolate =
        GradientLinearInterpolate::New();
      gi1Interpolate->SetInputImage(gi1);

      //ImageIO<GradientImage>::saveImage(gi1, "grad.mhd");

      //Region information of transformation
      ImageRegion region = i1->GetLargestPossibleRegion();
      ImageSize size = region.GetSize();
      ImageIndex index = region.GetIndex();


      //Iterators for the various images
      VectorType v;
      VectorType uv;
      VectorType vt;
      VectorType vp;


      transform->initIteration(region); 
      transformUpdate->initIteration(region); 

      GradientImageIterator gi1It(gi1, region);
      IndexedImageIterator i1It(i1, region);
      ImageIterator i2It(i2, region);
      ImageIterator maskIt(mask, region);

      
      //number of pixels in mask
      int nValid = 0;
      for(maskIt.GoToBegin(); !maskIt.IsAtEnd(); ++maskIt){
          if(maskIt.Get() != 0){
            nValid++;
          }
      };


      //Iterate until maximum number of iterations or 
      //stopping criteria is achieved   
      rmse =0; 
      int nIter = 0;     
      TPrecision energyPrev = std::numeric_limits<TPrecision>::max();

      TPrecision beta = 1.0-alpha;

      for(;nIter<nIterations; nIter++){

        //1. Compute Vector Field update
        TPrecision diffIntensity = 0;
        TPrecision maxUpdate = 0;
        TPrecision se = 0;
        TPrecision frob = 0;
        TPrecision mag = 0;

        transformUpdate->goToBegin();
        transform->goToBegin();
        i1It.GoToBegin();
        i2It.GoToBegin();
        gi1It.GoToBegin(); 
        maskIt.GoToBegin();
        for(; !maskIt.IsAtEnd(); ++i1It, ++i2It, ++gi1It, transform->next(),
            transformUpdate->next(), ++maskIt){

          if(maskIt.Get() != 0){

            //physical coordinates
            ImageIndex x = i1It.GetIndex();
            transform->get(v);
            
            Point p;
            i1->TransformIndexToPhysicalPoint(x, p); 
            
            ImageContinuousIndex cindexi2;
            i2->TransformPhysicalPointToContinuousIndex(p, cindexi2);
            
            transform->get(v);
            p += v;
            
            ImageContinuousIndex cindexg;
            gi1->TransformPhysicalPointToContinuousIndex(p, cindexg);

            ImageContinuousIndex cindexi1;
            i1->TransformPhysicalPointToContinuousIndex(p, cindexi1);
 
            for(unsigned int i=0; i<ImageIndex::GetIndexDimension(); i++){
              //check range of index
              if(cindexi1[i] < index[i]){
                cindexi1[i] = index[i];
              }
              else if(cindexi1[i] > index[i] + size[i] - 1){
                cindexi1[i] = index[i] + size[i] - 1;
              }
              if(cindexi2[i] < index[i]){
                cindexi2[i] = index[i];
              }
              else if(cindexi2[i] > index[i] + size[i] - 1){
                cindexi2[i] = index[i] + size[i] - 1;
              }
              if(cindexg[i] < index[i]){
                cindexg[i] = index[i];
              }
              else if(cindexg[i] > index[i] + size[i] - 1){
                cindexg[i] = index[i] + size[i] - 1;
              }
            };
            
            /*
            ImageContinuousIndex cindex;

            for(unsigned int i=0; i<ImageIndex::GetIndexDimension(); i++){
              cindex[i] = x[i] + v[i]/spacing[i];
              //check range of index
              if(cindex[i] < index[i]){
                cindex[i] = index[i];
              }
              else if(cindex[i] > index[i] + size[i] - 1){
                cindex[i] = index[i] + size[i] - 1;
              }
            };
            */
            



            //Gradient of i1 at x + v(x) 
            GradientInterpolateOutput g =
              gi1Interpolate->EvaluateAtContinuousIndex(cindexg);

            //Value of i1 at x + v(x)
            TPrecision i1w = i1Interpolate->EvaluateAtContinuousIndex(cindexi1);

            //value of i2 at x
            TPrecision i2o = i2Interpolate->EvaluateAtContinuousIndex(cindexi2);

            for(int i=0; i<dimension; i++){
              tmpVector[i] = (TPrecision) g[i];
            }
            diffIntensity = i1w - i2o; 
            se += diffIntensity * diffIntensity;
            tmpVector *= -(diffIntensity)*lambda;
            TPrecision tmpLength = tmpVector.GetSquaredNorm();
            if(tmpLength > maxUpdate){
              maxUpdate = tmpLength;
            }
            transformUpdate->set(  tmpVector  );
          }
        }

        rmse = sqrt(se/nValid);
        frob = transform->sumJacobianFrobeniusSquared();
        mag = transform->sumMagnitudeSquared();

        TPrecision energy = lambda * se + beta * mag + alpha*frob;
        energy = energy/nValid;
#ifdef VERBOSE
        std::cout << "iteration: " << nIter << std::endl;
        std::cout << "mse intensity: " << se/nValid << std::endl;
        std::cout << "energy: " << energy << std::endl << std::endl;
#endif
        //Check if we are within tolerance
        if( energy  < epsilon ){
          break;
        }
        if(energy!=energy){
          break;
        };

        if(energyPrev - energy <= lambdaIncThreshold){
          lambda += lambdaInc;
          if(lambdaInc == 0 ){
            break;
          }
          //maxMotion = 0;
          //maxUpdate = 0;
#ifdef VERBOSE
          std::cout << "Increasing lambda" << std::endl;
          std::cout << lambda << std::endl;
#endif
        }
        
  
        
        //2. Compute Time Step
        TPrecision dt =  maxMotion / ( sqrt(maxUpdate) + EPSILON ) * scaling ;
        TPrecision bdt = 1.0 / (1.0 + beta * dt);
        if(maxUpdate == 0){
          dt = 0;
          bdt = 1;
        }

        //if(energyPrev - energy > lambdaIncThreshold || dt == 0){
        //3. Apply update time step dt
        transformUpdate->goToBegin();
        transform->goToBegin();
        maskIt.GoToBegin();
        for(;!maskIt.IsAtEnd();  transform->next(),  
            transformUpdate->next(), ++maskIt){

          if(maskIt.Get()!=0){
            transform->get(v);
            transformUpdate->get(uv);
            v += (dt *uv);
            v *= bdt;

            /*for(unsigned int i=0; i< ImageIndex::GetIndexDimension(); i++){
              v[i] += dt *uv[i] * spacing[i];
              v[i] *= bdt * spacing[i];
            }*/
            transform->set(v); 
          }
        }
       // }

        if(dt == 0){
          dt = maxMotion;
        }
        //4. Smooth vectorfield
        TPrecision sigma = sqrt(2.0 * dt * alpha *bdt );

#ifdef VERBOSE
        std::cout << "dt: " << dt << std::endl;
        std::cout << "MaxUpdate: " << sqrt(maxUpdate) << std::endl;
        std::cout << "sigma: " << sigma << std::endl << std::endl;
#endif

        
        if(sigma!=0){
          transform->blur(sigma);
        }

        energyPrev = energy;
      }

#ifdef VERBOSE
      std::cout << "******nIterations: " << nIter << std::endl << std::endl<< std::endl;
#endif

      //delete transformTmp;
      delete transformUpdate;


      return transform;
    };



    TPrecision getRMSE(){
      return rmse;
    };



    //relative weight for laplacian vs. linear penalty
    void setAlpha(TPrecision a){
      if(a < 0 || a > 1){
        throw "alpha needs to be between 0 and 1";
      }
      alpha = a;
    };
    
    TPrecision getAlpha(){
      return alpha;
    };


    //Set lambda to start with
    void setLambda(TPrecision l){
      lambda = l;
    };

    TPrecision getLambda(){
      return lambda;
    };

    //stepwise increase in lambda
    void setLambdaIncrease(TPrecision inc){
      lambdaInc = inc;
    };
    
    TPrecision getLambdaIncrease(){
      return lambdaInc;
    };
    
    void setLambdaIncreaseThreshold(TPrecision incT){
      lambdaIncThreshold = incT;
    };
    
    TPrecision getLambdaIncreaseThreshold(){
      return lambdaIncThreshold;
    };



    //motion scaling per iteration
    void setMaximumMotion(TPrecision maxm){
      maxMotion = maxm;
    };

    //mx number of iterations
    void setMaximumIterations(int maxIter){
      nIterations = maxIter;
    };

    //Match images to within epsilon
    void setEpsilon(TPrecision eps){
      epsilon = eps;
    };

    TPrecision getEpsilon(){
      return epsilon;
    };


    //Create multiresolution images
    ImagePointer *downsample(ImagePointer im, int nres, TPrecision sigma){
      ImagePointer *pyramid = new ImagePointer[nres+1];
      pyramid[0] = im; 
      //Downsample
      for(int i=1; i<nres; i++){
        GaussianFilterPointer smooth = GaussianFilter::New();
        smooth->SetSigma(sigma);
        smooth->SetInput(pyramid[i-1]);
        smooth->Update();

        ResampleFilterPointer downsample = ResampleFilter::New();
        downsample->SetInput(smooth->GetOutput());
        downsample->SetOutputParametersFromImage(pyramid[i-1]);
                
        ImageSize size = pyramid[i-1]->GetLargestPossibleRegion().GetSize();
        ImageSpacing spacing = pyramid[i-1]->GetSpacing();
        for(unsigned int n = 0; n < size.GetSizeDimension(); n++){
          size[n] = size[n]/2;
          spacing[n] = spacing[n]*2;
        }
          
        downsample->SetOutputSpacing(spacing); 
        downsample->SetSize(size);

        downsample->Update();
        pyramid[i] = downsample->GetOutput();       
      }

      return pyramid;
    };

    void upsample(VImage *transform, ImageSize size, ImageSpacing spacing){
      //upsample transfrom
      ImagePointer *vfieldComps = transform->getComps();
      for(unsigned int j=0; j <  Image::GetImageDimension(); j++){
        ResampleFilterPointer upsample = ResampleFilter::New();
        upsample->SetInput(vfieldComps[j]);
        upsample->SetOutputParametersFromImage(vfieldComps[j]);

        
        upsample->SetOutputSpacing(spacing); 
        upsample->SetSize(size);
        

        upsample->Update();

	
        vfieldComps[j] = upsample->GetOutput();
        transform->setComp(j, vfieldComps[j]);
      }

    };

  private:


    typedef typename itk::Point<TPrecision, Image::ImageDimension > Point;

    typedef typename itk::GradientImageFilter<Image, TPrecision, TPrecision>
      GradientImageFilter;
    typedef typename GradientImageFilter::Pointer GradientImageFilterPointer;

    typedef typename GradientImageFilter::OutputImageType GradientImage;
    typedef typename GradientImage::Pointer GradientImagePointer;
    typedef typename GradientImage::PixelType GradientType;

    typedef typename itk::ImageRegionIterator<Image> ImageIterator;
    typedef typename itk::ImageRegionIteratorWithIndex<Image> IndexedImageIterator;
    typedef typename itk::ImageRegionIterator<GradientImage> GradientImageIterator;

    typedef typename itk::VectorLinearInterpolateImageFunction<GradientImage, TPrecision>
      GradientLinearInterpolate;
    typedef typename GradientLinearInterpolate::Pointer GradientLinearInterpolatePointer;
    typedef typename GradientLinearInterpolate::OutputType
      GradientInterpolateOutput;

    
    typedef typename itk::SmoothingRecursiveGaussianImageFilter<Image, Image> GaussianFilter;
    typedef typename GaussianFilter::Pointer GaussianFilterPointer;

    typedef typename itk::ResampleImageFilter<Image, Image> ResampleFilter;
    typedef typename ResampleFilter::Pointer ResampleFilterPointer;

    TPrecision maxMotion;
    int nIterations;
    TPrecision lambda;
    TPrecision lambdaInc;
    TPrecision lambdaIncThreshold;
    TPrecision epsilon;
    TPrecision alpha;
    TPrecision rmse;

};


#endif
