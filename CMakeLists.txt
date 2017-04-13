CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
IF(COMMAND CMAKE_POLICY)
  CMAKE_POLICY(SET CMP0003 NEW)
ENDIF(COMMAND CMAKE_POLICY)

PROJECT(MOTIONCORRECTION)

FIND_PACKAGE(ITK REQUIRED)
INCLUDE(${ITK_USE_FILE})


ADD_EXECUTABLE(GenerateTestVolume GenerateTestVolume.cxx)
TARGET_LINK_LIBRARIES (GenerateTestVolume ${ITK_LIBRARIES} )


ADD_EXECUTABLE(BandPassTransform BandPassTransform.cxx)
TARGET_LINK_LIBRARIES (BandPassTransform ${ITK_LIBRARIES} )

ADD_EXECUTABLE(EstimateMeanTransform FitNormal.cxx)
TARGET_LINK_LIBRARIES (EstimateMeanTransform ${ITK_LIBRARIES} )

ADD_EXECUTABLE(SliceToSliceAffine SliceToSliceAffine.cxx)
TARGET_LINK_LIBRARIES (SliceToSliceAffine ${ITK_LIBRARIES} )

ADD_EXECUTABLE(SliceToSliceAffine2 SliceToSliceAffine2.cxx)
TARGET_LINK_LIBRARIES (SliceToSliceAffine2 ${ITK_LIBRARIES} )

ADD_EXECUTABLE(SliceToSliceBSpline SliceToSliceBSpline.cxx)
TARGET_LINK_LIBRARIES (SliceToSliceBSpline ${ITK_LIBRARIES} )

ADD_EXECUTABLE(SliceToSmoothSliceBSpline SliceToSmoothSliceBSpline.cxx)
TARGET_LINK_LIBRARIES (SliceToSmoothSliceBSpline ${ITK_LIBRARIES} )

ADD_EXECUTABLE(SliceToSliceBSplinePlusCorrection SliceToSliceBSplinePlusCorrection.cxx)
TARGET_LINK_LIBRARIES (SliceToSliceBSplinePlusCorrection ${ITK_LIBRARIES} )


ADD_EXECUTABLE(SliceToSliceBSplineComposite SliceToSliceBSplineComposite.cxx)
TARGET_LINK_LIBRARIES (SliceToSliceBSplineComposite ${ITK_LIBRARIES} )

ADD_EXECUTABLE(SliceToSliceElastic SliceToSliceElastic.cxx)
TARGET_LINK_LIBRARIES (SliceToSliceElastic ${ITK_LIBRARIES} )


