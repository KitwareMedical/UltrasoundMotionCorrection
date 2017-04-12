#ifndef VECTORIMAGE_H
#define VECTORIMAGE_H

#include "itkVariableLengthVector.h"
#include "itkVector.h"
#include "itkImageRegionIterator.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkDerivativeImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkAddImageFilter.h"

#include "ImageIO.h"

//Helper class for SimpleFlow and SimpleWarp. Hack to have faster gaussian
//blurring for vectorfields.
template<typename TImage, unsigned int nComps>
class VectorImage{
  public:
    typedef TImage Image;
    typedef typename Image::Pointer ImagePointer;
    typedef typename Image::PixelType TPrecision;
    typedef typename Image::IndexType ImageIndex;
    typedef typename Image::RegionType ImageRegion;
    typedef typename Image::SizeType ImageSize;
    typedef typename Image::SpacingType ImageSpacing;

    typedef typename itk::Vector<TPrecision, nComps> VectorType;
    typedef typename itk::CovariantVector<TPrecision, nComps> CoVectorType;

    typedef typename itk::Image< CoVectorType, TImage::ImageDimension> ITKVectorImage;
    typedef typename ITKVectorImage::Pointer ITKVectorImagePointer;
    
    typedef typename itk::MultiplyImageFilter<Image, Image> MultiplyFilter;
    typedef typename MultiplyFilter::Pointer MultiplyFilterPointer;

    typedef typename itk::AddImageFilter<Image, Image> AddFilter;
    typedef typename AddFilter::Pointer AddFilterPointer;

    typedef typename itk::ImageRegionIterator<TImage> ImageIterator;    
    typedef typename itk::ImageRegionIterator<ITKVectorImage> ITKVectorImageIterator;

    //typedef typename itk::DiscreteGaussianImageFilter<TImage, TImage> GaussianFilter;
    typedef typename itk::SmoothingRecursiveGaussianImageFilter<TImage, TImage> GaussianFilter;
    typedef typename GaussianFilter::Pointer GaussianFilterPointer;
    typedef typename GaussianFilter::SigmaArrayType SigmaArray;
    

    //Derivative filter
    typedef typename itk::DerivativeImageFilter<TImage, TImage>
      DerivativeFilter;
    typedef typename DerivativeFilter::Pointer DerivativeFilterPointer;



    VectorImage(){
    };



    VectorImage(ITKVectorImagePointer im){
      
      for(unsigned int i=0; i< nComps; i++){
        comps[i] = TImage::New();
        comps[i]->SetRegions(im->GetLargestPossibleRegion());
        comps[i]->SetSpacing(im->GetSpacing());
        comps[i]->SetNumberOfComponentsPerPixel(im->GetNumberOfComponentsPerPixel());
        comps[i]->Allocate();
        ImageIterator it(comps[i], comps[i]->GetLargestPossibleRegion() );
      }
      initIteration(im->GetLargestPossibleRegion());
      ITKVectorImageIterator it(im, im->GetLargestPossibleRegion());
      for(; !it.IsAtEnd(); ++it, next()){
        CoVectorType v = it.Get();
        this->set(v);
      }

    };
    
    
    ~VectorImage(){
    };





    ImagePointer *getComps(){   
      return comps;
    };



    void setComp(int i, ImagePointer c){
      comps[i] = c;
    };
    

    void createZero(ImagePointer mold){
      for(unsigned int i=0; i< nComps; i++){
        comps[i] = TImage::New();
        comps[i]->SetRegions(mold->GetLargestPossibleRegion());
        comps[i]->SetSpacing(mold->GetSpacing());
        comps[i]->SetNumberOfComponentsPerPixel(mold->GetNumberOfComponentsPerPixel());
        comps[i]->Allocate();
        ImageIterator it(comps[i], comps[i]->GetLargestPossibleRegion() );
        for(it.GoToBegin(); !it.IsAtEnd(); ++it){
          it.Set(0);      
        }
      }
    };



    void initIteration(const ImageRegion &region){
      for(unsigned int i=0; i< nComps; i++){
        iters[i] = ImageIterator(comps[i], region);
      }
    };


    void goToBegin(){
      for(unsigned int i=0; i< nComps; i++){
        iters[i].GoToBegin();
      }
    };

    void get(VectorType &out){
      for(unsigned int i=0; i<out.Size() && i<nComps; i++){
        out[i] = iters[i].Get();
      }
    };

 
    void next(){
      for(unsigned int i=0; i<nComps; i++){
        ++iters[i];
      }
    };    
    

    void set(VectorType &in){
      for(unsigned int i=0; i<in.Size() && i<nComps; i++){
        iters[i].Set(in[i]);
      }
    };
    
    
    void set(CoVectorType &in){
      for(unsigned int i=0; i<in.Size() && i<nComps; i++){
        iters[i].Set(in[i]);
      }
    };
    
    
    
    bool isIterationAtEnd(){
      return iters[0].IsAtEnd();
    };

    



    void blur(TPrecision sigma, bool imagespacing  = true){
      for(unsigned int i=0; i<nComps; i++){
        GaussianFilterPointer blurFilter = GaussianFilter::New();
        //blurFilter[i]->SetVariance(sigma*sigma);
	//TODO deal with anistropic spacings
	TPrecision sigmaTmp = sigma;
	if( imagespacing ){
	  ImageSpacing spacing = comps[i]->GetSpacing();
          sigmaTmp *= spacing[0];
	}
	blurFilter->SetSigma( sigmaTmp );
        blurFilter->SetInput(comps[i]);
        blurFilter->Modified();
        blurFilter->Update();
        comps[i] = blurFilter->GetOutput();
      }
      initIteration(iters[0].GetRegion());
    };



    void blur(SigmaArray sigma, bool imagespacing  = true){
      ImageSpacing spacing = comps[0]->GetSpacing();
      if( imagespacing ){
	for(int i=0; i<sigma.Size(); i++){
          sigma[i] *= spacing[i];
	}
      }
      for(unsigned int i=0; i<nComps; i++){
        GaussianFilterPointer blurFilter = GaussianFilter::New();
        //blurFilter[i]->SetVariance(sigma*sigma);
	blurFilter->SetSigmaArray( sigma );
        blurFilter->SetInput(comps[i]);
        blurFilter->Modified();
        blurFilter->Update();
        comps[i] = blurFilter->GetOutput();
      }
      initIteration(iters[0].GetRegion());
    };


    void copy(VectorImage<Image, nComps> *image){
      for(unsigned int i=0; i<nComps; i++){
        comps[i] = ImageIO<Image>::copyImage(image->comps[i]);
      }
    };




    ITKVectorImagePointer toITK(){
      ITKVectorImagePointer im = ITKVectorImage::New();
      im->SetRegions(comps[0]->GetLargestPossibleRegion());
      im->SetSpacing(comps[0]->GetSpacing());
      im->SetNumberOfComponentsPerPixel(TImage::GetImageDimension());
      im->Allocate();
      ITKVectorImageIterator it(im, im->GetLargestPossibleRegion() );
      
      ImageIterator cIt[nComps];
      for(unsigned int i=0; i<nComps; i++){
        cIt[i] = ImageIterator(comps[i], comps[i]->GetLargestPossibleRegion());
      }

      for(it.GoToBegin(); !it.IsAtEnd(); ++it){
          CoVectorType v;
          for(unsigned int i=0; i<nComps; i++){
            v[i] = cIt[i].Get();
          }
          it.Set(v); 
          for(unsigned int i=0; i<nComps; i++){
            ++cIt[i];
          }
      }
      
      return im;
    };




    void add(VectorImage<TImage, nComps> *v){
      for(unsigned int i=0; i<nComps; i++){      
        AddFilterPointer addFilter = AddFilter::New();
        addFilter->SetInput1( comps[i] );
        addFilter->SetInput2( v->comps[i] );
        addFilter->Update();
	comps[i] = addFilter->GetOutput();
      }
    
    };
        

    
    void multiply(ImagePointer w){
      for(unsigned int i=0; i<nComps; i++){      
        MultiplyFilterPointer multiplyFilter = MultiplyFilter::New();
        multiplyFilter->SetInput1( comps[i] );
        multiplyFilter->SetInput2( w );
        multiplyFilter->Update();
	comps[i] = multiplyFilter->GetOutput();
      }
    };

        
    void multiply(TPrecision s){
      for(unsigned int i=0; i<nComps; i++){      
        MultiplyFilterPointer multiplyFilter = MultiplyFilter::New();
        multiplyFilter->SetInput( comps[i] );
        multiplyFilter->SetConstant( s );
        multiplyFilter->Update();
	comps[i] = multiplyFilter->GetOutput();
      }
    };



    ImagePointer weightedDivergence(ImagePointer weights){
      ImagePointer div = ImageIO<TImage>::copyImage(comps[0]);
     
      //dx
      ImagePointer ders[nComps];
      ImageIterator dersIt[nComps];
      for(unsigned int i=0; i<nComps; i++){      
        MultiplyFilterPointer multiplyFilter = MultiplyFilter::New();
        multiplyFilter->SetInput1( comps[i] );
        multiplyFilter->SetInput2( weights );

        DerivativeFilterPointer dFilter = DerivativeFilter::New();
        dFilter->SetOrder(1);
        dFilter->SetDirection(i);
        dFilter->SetInput( multiplyFilter->GetOutput() );
	dFilter->Update();
        ders[i] = dFilter->GetOutput();
        dersIt[i] = ImageIterator(ders[i], ders[i]->GetLargestPossibleRegion());
      }

      ImageIterator it(div, div->GetLargestPossibleRegion());
      for(; !it.IsAtEnd(); ++it){
        TPrecision tmp = 0;
        for(unsigned int i=0; i<nComps; i++){
          tmp += dersIt[i].Get();
          ++(dersIt[i]);
        }
        it.Set(tmp);
      }
      return div;
    };





    ImagePointer divergence(){
      ImagePointer div = ImageIO<TImage>::createImage(comps[0]);
     
      //dx
      ImagePointer ders[nComps];
      ImageIterator dersIt[nComps];
      for(unsigned int i=0; i<nComps; i++){
        DerivativeFilterPointer dFilter = DerivativeFilter::New();
        dFilter->SetOrder(1);
        dFilter->SetDirection(i);
        dFilter->SetInput(comps[i]);
        dFilter->Update();
        ders[i] = dFilter->GetOutput();
        dersIt[i] = ImageIterator(ders[i], ders[i]->GetLargestPossibleRegion());
      }

      ImageIterator it(div, div->GetLargestPossibleRegion());
      for(; !it.IsAtEnd(); ++it){
        TPrecision tmp = 0;
        for(unsigned int i=0; i<nComps; i++){
          tmp += dersIt[i].Get();
          ++(dersIt[i]);
        }
        it.Set(tmp);
      }
      return div;
    };

    TPrecision sumDivergence(){
      ImagePointer div = divergence();
      ImageIterator it(div, div->GetLargestPossibleRegion());
      TPrecision sum = 0;
      for(; !it.IsAtEnd(); ++it){
        sum += it.Get();
      }
      return sum;
    };



    ImagePointer jacobianFrobeniusSquared(){
      using namespace FortranLinalg;
      ImagePointer ders[nComps][Image::ImageDimension];
      ImageIterator dersIt[nComps][Image::ImageDimension];

      for(unsigned int i=0; i<nComps; i++){
        for(unsigned int j=0; j<Image::ImageDimension; j++){
          ders[i][j] = derivative(i, j);
          dersIt[i][j] = ImageIterator(ders[i][j], ders[i][j]->GetLargestPossibleRegion());
        }
      }
      
      ImagePointer jnorm = ImageIO<TImage>::copyImage(comps[0]);
      ImageIterator it(jnorm, jnorm->GetLargestPossibleRegion());

      DenseMatrix<TPrecision> jac(nComps, Image::ImageDimension);
      for(;!it.IsAtEnd(); ++it){
        TPrecision tmp;
        TPrecision norm = 0;
        for(unsigned int i=0; i<nComps; i++){
          for(unsigned int j=0; j<Image::ImageDimension; j++){
            tmp = dersIt[i][j].Get();
            norm += tmp*tmp;
            ++(dersIt[i][j]);
          }
        }
        it.Set(norm); 
      }

      return jnorm; 
    };



    TPrecision sumJacobianFrobeniusSquared(){
      ImagePointer jf = jacobianFrobeniusSquared();
      ImageIterator it(jf, jf->GetLargestPossibleRegion());
      TPrecision sum = 0;
      for(; !it.IsAtEnd(); ++it){
        sum += it.Get();
      }
      return sum;
    };



    ImagePointer derivative(int dim, int component){
      DerivativeFilterPointer dFilter = DerivativeFilter::New();
      dFilter->SetOrder(1);
      dFilter->SetDirection(dim);
      dFilter->SetInput(comps[component]);
      dFilter->Update();
      ImagePointer der = dFilter->GetOutput();
      return der;
    };





    ImagePointer magnitudeSquared(){
      ImagePointer mag = ImageIO<TImage>::copyImage(comps[0]);
      initIteration(mag->GetLargestPossibleRegion());
      ImageIterator it(mag, mag->GetLargestPossibleRegion());
      VectorType v;
      for(;!it.IsAtEnd(); ++it, next()){
        get(v);
        it.Set(v.GetSquaredNorm());  
      }

      return mag;
    };


    TPrecision sumMagnitudeSquared(){
      ImagePointer mag = magnitudeSquared();
      ImageIterator it(mag, mag->GetLargestPossibleRegion());
      TPrecision sum = 0;
      for(; !it.IsAtEnd(); ++it){
        sum += it.Get();
      }
      return sum;
    };


  private:

    ImagePointer comps[nComps];
    ImageIterator iters[nComps];


};

#endif


