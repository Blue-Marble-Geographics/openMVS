/*
* DepthMap.cpp
*
* Copyright (c) 2014-2015 SEACAVE
*
* Author(s):
*
*      cDc <cdc.seacave@gmail.com>
*
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU Affero General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Affero General Public License for more details.
*
* You should have received a copy of the GNU Affero General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*
* Additional Terms:
*
*      You are required to preserve legal notices and author attributions in
*      that material or in the Appropriate Legal Notices displayed by works
*      containing it.
*/

#include "Common.h"
#include "DepthMap.h"
#include "Mesh.h"
#define _USE_OPENCV
#include "Interface.h"
#include "../Common/AutoEstimator.h"
// CGAL: depth-map initialization
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
// CGAL: estimate normals
#include <CGAL/Simple_cartesian.h>
#include <CGAL/property_map.h>
#include <CGAL/pca_estimate_normals.h>

using namespace MVS;


// D E F I N E S ///////////////////////////////////////////////////

#define DEFVAR_OPTDENSE_string(name, title, desc, ...)  DEFVAR_string(OPTDENSE, name, title, desc, __VA_ARGS__)
#define DEFVAR_OPTDENSE_bool(name, title, desc, ...)    DEFVAR_bool(OPTDENSE, name, title, desc, __VA_ARGS__)
#define DEFVAR_OPTDENSE_int32(name, title, desc, ...)   DEFVAR_int32(OPTDENSE, name, title, desc, __VA_ARGS__)
#define DEFVAR_OPTDENSE_uint32(name, title, desc, ...)  DEFVAR_uint32(OPTDENSE, name, title, desc, __VA_ARGS__)
#define DEFVAR_OPTDENSE_flags(name, title, desc, ...)   DEFVAR_flags(OPTDENSE, name, title, desc, __VA_ARGS__)
#define DEFVAR_OPTDENSE_float(name, title, desc, ...)   DEFVAR_float(OPTDENSE, name, title, desc, __VA_ARGS__)
#define DEFVAR_OPTDENSE_double(name, title, desc, ...)  DEFVAR_double(OPTDENSE, name, title, desc, __VA_ARGS__)

#define MDEFVAR_OPTDENSE_string(name, title, desc, ...) DEFVAR_string(OPTDENSE, name, title, desc, __VA_ARGS__)
#define MDEFVAR_OPTDENSE_bool(name, title, desc, ...)   DEFVAR_bool(OPTDENSE, name, title, desc, __VA_ARGS__)
#define MDEFVAR_OPTDENSE_int32(name, title, desc, ...)  DEFVAR_int32(OPTDENSE, name, title, desc, __VA_ARGS__)
#define MDEFVAR_OPTDENSE_uint32(name, title, desc, ...) DEFVAR_uint32(OPTDENSE, name, title, desc, __VA_ARGS__)
#define MDEFVAR_OPTDENSE_flags(name, title, desc, ...)  DEFVAR_flags(OPTDENSE, name, title, desc, __VA_ARGS__)
#define MDEFVAR_OPTDENSE_float(name, title, desc, ...)  DEFVAR_float(OPTDENSE, name, title, desc, __VA_ARGS__)
#define MDEFVAR_OPTDENSE_double(name, title, desc, ...) DEFVAR_double(OPTDENSE, name, title, desc, __VA_ARGS__)

namespace MVS {
DEFOPT_SPACE(OPTDENSE, _T("Dense"))

DEFVAR_OPTDENSE_uint32(nResolutionLevel, "Resolution Level", "How many times to scale down the images before dense reconstruction", "1")
MDEFVAR_OPTDENSE_uint32(nMaxResolution, "Max Resolution", "Do not scale images lower than this resolution", "3200")
MDEFVAR_OPTDENSE_uint32(nMinResolution, "Min Resolution", "Do not scale images lower than this resolution", "640")
DEFVAR_OPTDENSE_uint32(nSubResolutionLevels, "SubResolution levels", "Number of lower resolution levels to estimate the depth and normals", "2")
DEFVAR_OPTDENSE_uint32(nMinViews, "Min Views", "minimum number of agreeing views to validate a depth", "2")
MDEFVAR_OPTDENSE_uint32(nMaxViews, "Max Views", "maximum number of neighbor images used to compute the depth-map for the reference image", "12")
DEFVAR_OPTDENSE_uint32(nMinViewsFuse, "Min Views Fuse", "minimum number of images that agrees with an estimate during fusion in order to consider it inlier (<2 - only merge depth-maps)", "2")
DEFVAR_OPTDENSE_uint32(nMinViewsFilter, "Min Views Filter", "minimum number of images that agrees with an estimate in order to consider it inlier", "2")
MDEFVAR_OPTDENSE_uint32(nMinViewsFilterAdjust, "Min Views Filter Adjust", "minimum number of images that agrees with an estimate in order to consider it inlier (0 - disabled)", "1")
MDEFVAR_OPTDENSE_uint32(nMinViewsTrustPoint, "Min Views Trust Point", "min-number of views so that the point is considered for approximating the depth-maps (<2 - random initialization)", "2")
MDEFVAR_OPTDENSE_uint32(nNumViews, "Num Views", "Number of views used for depth-map estimation (0 - all views available)", "0", "1", "4")
MDEFVAR_OPTDENSE_uint32(nPointInsideROI, "Point Inside ROI", "consider a point shared only if inside ROI when estimating the neighbor views (0 - ignore ROI, 1 - weight more ROI points, 2 - consider only ROI points)", "1")
MDEFVAR_OPTDENSE_bool(bFilterAdjust, "Filter Adjust", "adjust depth estimates during filtering", "1")
MDEFVAR_OPTDENSE_bool(bAddCorners, "Add Corners", "add support points at image corners with nearest neighbor disparities", "0")
MDEFVAR_OPTDENSE_bool(bInitSparse, "Init Sparse", "init depth-map only with the sparse points (no interpolation)", "1")
MDEFVAR_OPTDENSE_bool(bRemoveDmaps, "Remove Dmaps", "remove depth-maps after fusion", "0")
MDEFVAR_OPTDENSE_float(fViewMinScore, "View Min Score", "Min score to consider a neighbor images (0 - disabled)", "2.0")
MDEFVAR_OPTDENSE_float(fViewMinScoreRatio, "View Min Score Ratio", "Min score ratio to consider a neighbor images", "0.03")
MDEFVAR_OPTDENSE_float(fMinArea, "Min Area", "Min shared area for accepting the depth triangulation", "0.05")
MDEFVAR_OPTDENSE_float(fMinAngle, "Min Angle", "Min angle for accepting the depth triangulation", "3.0")
MDEFVAR_OPTDENSE_float(fOptimAngle, "Optim Angle", "Optimal angle for computing the depth triangulation", "12.0")
MDEFVAR_OPTDENSE_float(fMaxAngle, "Max Angle", "Max angle for accepting the depth triangulation", "65.0")
MDEFVAR_OPTDENSE_float(fDescriptorMinMagnitudeThreshold, "Descriptor Min Magnitude Threshold", "minimum patch texture variance accepted when matching two patches (0 - disabled)", "0.02") // 0.02: pixels with patch texture variance below 0.0004 (0.02^2) will be removed from depthmap; 0.12: patch texture variance below 0.02 (0.12^2) is considered texture-less
MDEFVAR_OPTDENSE_float(fDepthDiffThreshold, "Depth Diff Threshold", "maximum variance allowed for the depths during refinement", "0.01")
MDEFVAR_OPTDENSE_float(fNormalDiffThreshold, "Normal Diff Threshold", "maximum variance allowed for the normal during fusion (degrees)", "25")
MDEFVAR_OPTDENSE_float(fPairwiseMul, "Pairwise Mul", "pairwise cost scale to match the unary cost", "0.3")
MDEFVAR_OPTDENSE_float(fOptimizerEps, "Optimizer Eps", "MRF optimizer stop epsilon", "0.001")
MDEFVAR_OPTDENSE_int32(nOptimizerMaxIters, "Optimizer Max Iters", "MRF optimizer max number of iterations", "80")
MDEFVAR_OPTDENSE_uint32(nSpeckleSize, "Speckle Size", "maximal size of a speckle (small speckles get removed)", "100")
MDEFVAR_OPTDENSE_uint32(nIpolGapSize, "Interpolate Gap Size", "interpolate small gaps (left<->right, top<->bottom)", "7")
MDEFVAR_OPTDENSE_int32(nIgnoreMaskLabel, "Ignore Mask Label", "label id used during ignore mask filter (<0 - disabled)", "-1")
DEFVAR_OPTDENSE_uint32(nOptimize, "Optimize", "should we filter the extracted depth-maps?", "7") // see DepthFlags
MDEFVAR_OPTDENSE_uint32(nEstimateColors, "Estimate Colors", "should we estimate the colors for the dense point-cloud?", "2", "0", "1")
MDEFVAR_OPTDENSE_uint32(nEstimateNormals, "Estimate Normals", "should we estimate the normals for the dense point-cloud?", "0", "1", "2")
MDEFVAR_OPTDENSE_float(fNCCThresholdKeep, "NCC Threshold Keep", "Maximum 1-NCC score accepted for a match", "0.9", "0.5")
DEFVAR_OPTDENSE_uint32(nEstimationIters, "Estimation Iters", "Number of patch-match iterations", "3")
DEFVAR_OPTDENSE_uint32(nEstimationGeometricIters, "Estimation Geometric Iters", "Number of geometric consistent patch-match iterations (0 - disabled)", "2")
MDEFVAR_OPTDENSE_float(fEstimationGeometricWeight, "Estimation Geometric Weight", "pairwise geometric consistency cost weight", "0.1")
MDEFVAR_OPTDENSE_uint32(nRandomIters, "Random Iters", "Number of iterations for random assignment per pixel", "6")
MDEFVAR_OPTDENSE_uint32(nRandomMaxScale, "Random Max Scale", "Maximum number of iterations to skip during random assignment", "2")
MDEFVAR_OPTDENSE_float(fRandomDepthRatio, "Random Depth Ratio", "Depth range ratio of the current estimate for random plane assignment", "0.003", "0.004")
MDEFVAR_OPTDENSE_float(fRandomAngle1Range, "Random Angle1 Range", "Angle 1 range for random plane assignment (degrees)", "16.0", "20.0")
MDEFVAR_OPTDENSE_float(fRandomAngle2Range, "Random Angle2 Range", "Angle 2 range for random plane assignment (degrees)", "10.0", "12.0")
MDEFVAR_OPTDENSE_float(fRandomSmoothDepth, "Random Smooth Depth", "Depth variance used during neighbor smoothness assignment (ratio)", "0.02")
MDEFVAR_OPTDENSE_float(fRandomSmoothNormal, "Random Smooth Normal", "Normal variance used during neighbor smoothness assignment (degrees)", "13")
MDEFVAR_OPTDENSE_float(fRandomSmoothBonus, "Random Smooth Bonus", "Score factor used to encourage smoothness (1 - disabled)", "0.93")
}



// S T R U C T S ///////////////////////////////////////////////////

//constructor from reference of DepthData
DepthData::DepthData(const DepthData& srcDepthData) :
	images(srcDepthData.images),
	neighbors(srcDepthData.neighbors),
	points(srcDepthData.points),
	mask(srcDepthData.mask),
	depthMap(srcDepthData.depthMap),
	normalMap(srcDepthData.normalMap),
	confMap(srcDepthData.confMap),
	dMin(srcDepthData.dMin),
	dMax(srcDepthData.dMax),
	references(srcDepthData.references)
{}

// return normal in world-space for the given pixel
// the 3D points can be precomputed and passed here
void DepthData::GetNormal(const ImageRef& ir, Point3f& N, const TImage<Point3f>* pPointMap) const
{
	ASSERT(!IsEmpty());
	ASSERT(depthMap(ir) > 0);
	const Camera& camera = images.First().camera;
	if (!normalMap.empty()) {
		// set available normal
		N = camera.R.t()*Cast<REAL>(normalMap(ir));
		return;
	}
	// estimate normal based on the neighbor depths
	const int nPointsStep = 2;
	const int nPointsHalf = 2;
	const int nPoints = 2*nPointsHalf+1;
	const int nWindowHalf = nPointsHalf*nPointsStep;
	const int nWindow = 2*nWindowHalf+1;
	const Image8U::Size size(depthMap.size());
	const ImageRef ptCorner(ir.x-nWindowHalf, ir.y-nWindowHalf);
	const ImageRef ptCornerRel(ptCorner.x>=0?0:-ptCorner.x, ptCorner.y>=0?0:-ptCorner.y);
	Point3Arr points(1, nPoints*nPoints);
	if (pPointMap) {
		points[0] = (*pPointMap)(ir);
		for (int j=ptCornerRel.y; j<nWindow; j+=nPointsStep) {
			const int y = ptCorner.y+j;
			if (y >= size.height)
				break;
			for (int i=ptCornerRel.x; i<nWindow; i+=nPointsStep) {
				const int x = ptCorner.x+i;
				if (x >= size.width)
					break;
				if (x==ir.x && y==ir.y)
					continue;
				if (depthMap(y,x) > 0)
					points.Insert((*pPointMap)(y,x));
			}
		}
	} else {
		points[0] = camera.TransformPointI2C(Point3(ir.x,ir.y,depthMap(ir)));
		for (int j=ptCornerRel.y; j<nWindow; j+=nPointsStep) {
			const int y = ptCorner.y+j;
			if (y >= size.height)
				break;
			for (int i=ptCornerRel.x; i<nWindow; i+=nPointsStep) {
				const int x = ptCorner.x+i;
				if (x >= size.width)
					break;
				if (x==ir.x && y==ir.y)
					continue;
				const Depth d = depthMap(y,x);
				if (d > 0)
					points.Insert(camera.TransformPointI2C(Point3(x,y,d)));
			}
		}
	}
	if (points.GetSize() < 3) {
		N = normalized(-points[0]);
		return;
	}
	Plane plane;
	if (EstimatePlaneThLockFirstPoint(points, plane, 0, NULL, 20) < 3) {
		N = normalized(-points[0]);
		return;
	}
	ASSERT(ISEQUAL(plane.m_vN.norm(),REAL(1)));
	// normal is defined up to sign; pick the direction that points to the camera
	if (plane.m_vN.dot((const Point3::EVec)points[0]) > 0)
		plane.Negate();
	N = camera.R.t()*Point3(plane.m_vN);
}
void DepthData::GetNormal(const Point2f& pt, Point3f& N, const TImage<Point3f>* pPointMap) const
{
	const ImageRef ir(ROUND2INT(pt));
	GetNormal(ir, N, pPointMap);
} // GetNormal
/*----------------------------------------------------------------*/


// apply mask to the depth map
void DepthData::ApplyIgnoreMask(const BitMatrix& mask)
{
	ASSERT(IsValid() && !IsEmpty() && mask.size() == depthMap.size());
	for (int r=0; r<depthMap.rows; ++r) {
		for (int c=0; c<depthMap.cols; ++c) {
			if (mask.isSet(r,c))
				continue;
			// discard depth-map section ignored by mask
			depthMap(r,c) = 0;
			if (!normalMap.empty())
				normalMap(r,c) = Normal::ZERO;
			if (!confMap.empty())
				confMap(r,c) = 0;
		}
	}
} // ApplyIgnoreMask
/*----------------------------------------------------------------*/


bool DepthData::Save(const String& fileName) const
{
	ASSERT(IsValid() && !depthMap.empty() && !confMap.empty());
	const String fileNameTmp(fileName+".tmp"); {
		// serialize out the current state
		IIndexArr IDs(0, images.size());
		for (const ViewData& image: images)
			IDs.push_back(image.GetID());
		const ViewData& image0 = GetView();
		if (!ExportDepthDataRaw(fileNameTmp, image0.pImageData->name, IDs, depthMap.size(), image0.camera.K, image0.camera.R, image0.camera.C, dMin, dMax, depthMap, normalMap, confMap, viewsMap))
			return false;
	}
	if (!File::renameFile(fileNameTmp, fileName)) {
		DEBUG_EXTRA("error: can not access dmap file '%s'", fileName.c_str());
		File::deleteFile(fileNameTmp);
		return false;
	}
	return true;
}
bool DepthData::Load(const String& fileName, unsigned flags)
{
	// serialize in the saved state
	String imageFileName;
	IIndexArr IDs;
	cv::Size imageSize;
	Camera camera;
	if (!ImportDepthDataRaw(fileName, imageFileName, IDs, imageSize, camera.K, camera.R, camera.C, dMin, dMax, depthMap, normalMap, confMap, viewsMap, flags))
		return false;
	ASSERT(!IsValid() || (IDs.size() == images.size() && IDs.front() == GetView().GetID()));
	ASSERT(depthMap.size() == imageSize);
	return true;
}
/*----------------------------------------------------------------*/


unsigned DepthData::GetRef()
{
	Lock l(cs);
	return references;
}
unsigned DepthData::IncRef(const String& fileName)
{
	Lock l(cs);
	ASSERT(!IsEmpty() || references==0);
	if (IsEmpty() && !Load(fileName))
		return 0;
	return ++references;
}
unsigned DepthData::DecRef()
{
	Lock l(cs);
	ASSERT(references>0);
	if (--references == 0)
		Release();
	return references;
}
/*----------------------------------------------------------------*/



// S T R U C T S ///////////////////////////////////////////////////

// try to load and apply mask to the depth map;
// the mask marks as false pixels that should be ignored
bool DepthEstimator::ImportIgnoreMask(const Image& image0, const Image8U::Size& size, BitMatrix& bmask, uint16_t nIgnoreMaskLabel)
{
	ASSERT(image0.IsValid() && !image0.image.empty());
	if (image0.maskName.empty())
		return false;
	Image16U mask;
	if (!mask.Load(image0.maskName)) {
		DEBUG("warning: can not load the segmentation mask '%s'", image0.maskName.c_str());
		return false;
	}
	cv::resize(mask, mask, size, 0, 0, cv::INTER_NEAREST);
	bmask.create(size);
	bmask.memset(0xFF);
	for (int r=0; r<size.height; ++r) {
		for (int c=0; c<size.width; ++c) {
			if (mask(r,c) == nIgnoreMaskLabel)
				bmask.unset(r,c);
		}
	}
	return true;
} // ImportIgnoreMask

// create the map for converting index to matrix position
//                         1 2 3
//  1 2 4 7 5 3 6 8 9 -->  4 5 6
//                         7 8 9
void DepthEstimator::MapMatrix2ZigzagIdx(const Image8U::Size& size, DepthEstimator::MapRefArr& coords, const BitMatrix& mask, int rawStride)
{
	typedef DepthEstimator::MapRef MapRef;
	const int w = size.width;
	const int w1 = size.width-1;
	coords.Empty();
	coords.Reserve(size.area());
	for (int dy=0, h=rawStride; dy<size.height; dy+=h) {
		if (h*2 > size.height - dy)
			h = size.height - dy;
		int lastX = 0;
		MapRef x(MapRef::ZERO);
		for (int i=0, ei=w*h; i<ei; ++i) {
			const MapRef pt(x.x, x.y+dy);
			if (mask.empty() || mask.isSet(pt))
				coords.Insert(pt);
			if (x.x-- == 0 || ++x.y == h) {
				if (++lastX < w) {
					x.x = lastX;
					x.y = 0;
				} else {
					x.x = w1;
					x.y = lastX - w1;
				}
			}
		}
	}
}

// replace POWI(0.5f, idxScaleRange):           0    1      2       3       4         5         6           7           8             9             10              11
const float DepthEstimator::scaleRanges[12] = {1.f, 0.5f, 0.25f, 0.125f, 0.0625f, 0.03125f, 0.015625f, 0.0078125f, 0.00390625f, 0.001953125f, 0.0009765625f, 0.00048828125f};

DepthEstimator::DepthEstimator(
	unsigned nIter, DepthData& _depthData0, volatile Thread::safe_t& _idx,
	#if DENSE_NCC == DENSE_NCC_WEIGHTED
	WeightMap& _weightMap0,
	#else
	const Image64F& _image0Sum,
	#endif
	const MapRefArr& _coords)
	:
	#ifndef _RELEASE
	rnd(SEACAVE::Random::default_seed),
	#endif
	idxPixel(_idx),
	neighbors(0,2),
	scores(_depthData0.images.size()-1),
	depthMap0(_depthData0.depthMap), normalMap0(_depthData0.normalMap), confMap0(_depthData0.confMap),
	#if DENSE_NCC == DENSE_NCC_WEIGHTED
	weightMap0(_weightMap0),
	#endif
	nIteration(nIter),
	images(_depthData0.images.begin()+1, _depthData0.images.end()), image0(_depthData0.images[0]),
	#if DENSE_NCC != DENSE_NCC_WEIGHTED
	image0Sum(_image0Sum),
	#endif
	coords(_coords), size(_depthData0.images.First().image.size()),
	dMin(_depthData0.dMin), dMax(_depthData0.dMax),
	dMinSqr(SQRT(_depthData0.dMin)), dMaxSqr(SQRT(_depthData0.dMax)),
	dir(nIter%2 ? RB2LT : LT2RB),
	#if DENSE_AGGNCC == DENSE_AGGNCC_NTH
	idxScore((_depthData0.images.size()-1)/3),
	#elif DENSE_AGGNCC == DENSE_AGGNCC_MINMEAN
	idxScore(_depthData0.images.size()<=2 ? 0u : 1u),
	#endif
	smoothBonusDepth(1.f-OPTDENSE::fRandomSmoothBonus), smoothBonusNormal((1.f-OPTDENSE::fRandomSmoothBonus)*0.96f),
	smoothSigmaDepth(-1.f/(2.f*SQUARE(OPTDENSE::fRandomSmoothDepth))), // used in exp(-x^2 / (2*(0.02^2)))
	smoothSigmaNormal(-1.f/(2.f*SQUARE(FD2R(OPTDENSE::fRandomSmoothNormal)))), // used in exp(-x^2 / (2*(0.22^2)))
	thMagnitudeSq(OPTDENSE::fDescriptorMinMagnitudeThreshold>0?SQUARE(OPTDENSE::fDescriptorMinMagnitudeThreshold):-1.f),
	angle1Range(FD2R(OPTDENSE::fRandomAngle1Range)), //default 0.279252678=FD2R(20)
	angle2Range(FD2R(OPTDENSE::fRandomAngle2Range)), //default 0.174532920=FD2R(16)
	thConfSmall(OPTDENSE::fNCCThresholdKeep * 0.66f), // default 0.6
	thConfBig(OPTDENSE::fNCCThresholdKeep * 0.9f), // default 0.8
	thConfRand(OPTDENSE::fNCCThresholdKeep * 1.1f), // default 0.99
	thRobust(OPTDENSE::fNCCThresholdKeep * 4.f / 3.f) // default 1.2
	#if DENSE_REFINE == DENSE_REFINE_EXACT
	, thPerturbation(1.f/POW(2.f,float(nIter+1)))
	#endif
{
	ASSERT(_depthData0.images.size() >= 1);
	if (
		(4 != GROUP_SIZE)
		|| (DENSE_NCC != DENSE_NCC_WEIGHTED)) {
		throw std::runtime_error( "Unsupported" );
	}
}

// center a patch of given size on the segment
bool DepthEstimator::PreparePixelPatch(const ImageRef& x)
{
	x0 = x;
	pWeightMap = &weightMap0[x0.y*image0.image.width()+x0.x];
	x0ULPatchCorner = Vec2_t((Calc_t)(x0.x-nSizeHalfWindow), (Calc_t) (x0.y-nSizeHalfWindow));
	return image0.image.isInside(ImageRef(x.x-nSizeHalfWindow, x.y-nSizeHalfWindow)) &&
	       image0.image.isInside(ImageRef(x.x+nSizeHalfWindow, x.y+nSizeHalfWindow));
}
// fetch the patch pixel values in the main image
bool DepthEstimator::FillPixelPatch()
{
	#if DENSE_NCC != DENSE_NCC_WEIGHTED
	const float mean(GetImage0Sum(x0)/nTexels);
	normSq0 = 0;
	float* pTexel0 = texels0.data();
	for (int i=-nSizeHalfWindow; i<=nSizeHalfWindow; i+=nSizeStep)
		for (int j=-nSizeHalfWindow; j<=nSizeHalfWindow; j+=nSizeStep)
			normSq0 += SQUARE(*pTexel0++ = image0.image(x0.y+i, x0.x+j)-mean);
	#else
	Weight& w = *pWeightMap;
	if (w.normSq0 == 0) {
		w.sumWeights = 0;
		int n = 0;
		const float colCenter = image0.image(x0);
		for (int i=-nSizeHalfWindow; i<=nSizeHalfWindow; i+=nSizeStep) {
			for (int j=-nSizeHalfWindow; j<=nSizeHalfWindow; j+=nSizeStep) {
				w.normSq0 +=
					(w.pixelTempWeights[n] = image0.image(x0.y + i, x0.x + j)) *
					(w.pixelWeights[n] = GetWeight(ImageRef(j, i), colCenter));
				w.sumWeights += w.pixelWeights[n];
				++n;
			}
		}

		ASSERT(n == nTexels);
		const float tm(w.normSq0/w.sumWeights);
		w.normSq0 = 0;
		n = 0;
		do {
			const float t(w.pixelTempWeights[n] - tm);
			w.normSq0 += (w.pixelTempWeights[n] = w.pixelWeights[n] * t) * t;
		} while (++n < nTexels);
	}
	normSq0 = w.normSq0;
	#endif
	if (normSq0 < thMagnitudeSq && (lowResDepthMap.empty() || lowResDepthMap(x0) <= 0))
		return false;
	X0 = image0.camera.TransformPointI2C(Cast<REAL>(x0));
	return true;
}

bool DepthEstimator::IsScorable(
	const DepthData::ViewData& image1,
	ScoreHelper& sh
)
{
	// Calculate H: ((image1.Hl + image1.Hm * mm) * image1.Hr)
	// Points are culled frequently (>50% in limited testing).
	// Attempt to identify them quickly.
	// Calculate just the first two rows of H.
	const Calc_t a = image1.Hl(0,0) + image1.Hm(0) * sh.mMat(0);
	const Calc_t j = image1.Hr(0,0);
	const Calc_t h00 = (Calc_t) (a * j);

	const Calc_t b = image1.Hl(0,1) + image1.Hm(0) * sh.mMat(1);
	const Calc_t l = image1.Hr(1,1);
	const Calc_t h10 = (Calc_t) (b * l);

	const Calc_t k = image1.Hr(0,2);
	const Calc_t m = image1.Hr(1,2);
	const Calc_t c = image1.Hl(0,2) + image1.Hm(0) * sh.mMat(2);
	const Calc_t h20 = (Calc_t) (a * k + b * m + c);

	const Calc_t g = image1.Hl(2,0) + image1.Hm(2) * sh.mMat(0);
	const Calc_t h = image1.Hl(2,1) + image1.Hm(2) * sh.mMat(1);
	const Calc_t i = image1.Hl(2,2) + image1.Hm(2) * sh.mMat(2);
	const Calc_t h02 = (Calc_t) (g * j);
	const Calc_t h12 = (Calc_t) (h * l);
	const Calc_t h22 = (Calc_t) (g * k + h * m + i);

	const Calc_t x = h00 * x0ULPatchCorner[0] + h10 * x0ULPatchCorner[1] + h20;
	const Calc_t z = h02 * x0ULPatchCorner[0] + h12 * x0ULPatchCorner[1] + h22;

	const Calc_t imageWidthWithBorder = (Calc_t) (image1.image.width() - 2);

	// Is the point culled horizontally?
	if ((x < 1. * z) | ((x > imageWidthWithBorder*z))) { // Binary | intentional
		return false;
	}

	// Compute the remaining row of H.
	const Calc_t d = image1.Hl(1,0) + image1.Hm(1) * sh.mMat(0);
	const Calc_t e = image1.Hl(1,1) + image1.Hm(1) * sh.mMat(1);
	const Calc_t f = image1.Hl(1,2) + image1.Hm(1) * sh.mMat(2);
	const Calc_t h01 = (Calc_t) (d * j);
	const Calc_t h11 = (Calc_t) (e * l);
	const Calc_t h21 = (Calc_t) (d * k + e * m + f);

	const Calc_t y = h01 * x0ULPatchCorner[0] + h11 * x0ULPatchCorner[1] + h21;

	const Calc_t imageHeightWithBorder = (Calc_t) (image1.image.height() - 2);

	// Is the point culled vertically?
	if ((y < 1. * z) | (y > imageHeightWithBorder*z)) { // Binary | intentional
		return false;
	}

	sh.mVX = _SetN((float) x, (float) y, (float) z, 0.f);

	// Prepare H for rasterizing.
	// Determine H's horizontal and vertical basis vectors.
	const _Data vHTRow0 = _SetN(h00, h01, h02, 0.f);
	const _Data vHTRow1 = _SetN(h10, h11, h12, 0.f);
	constexpr _Data vSizeStep = {(float) nSizeStep, (float) nSizeStep, (float) nSizeStep, (float) nSizeStep};
	sh.mVBasisH = _Mul(vHTRow0, vSizeStep);
	sh.mVBasisV = _Mul(vHTRow1, vSizeStep);
	// Epsilon is needed as a+a+a+a isn't necessarily equal to 4*a
	constexpr _Data vFourEpsilon = {4.f + FLT_EPSILON, 4.f + FLT_EPSILON, 4.f + FLT_EPSILON, 0.f};
	const _Data vBasisH4 = _Mul(sh.mVBasisH, vFourEpsilon);
	const _Data vBasisV4 = _Mul(sh.mVBasisV, vFourEpsilon);

	sh.mVImageWidthWithBorder = _Set((float) imageWidthWithBorder);
	sh.mVImageHeightWithBorder = _Set((float) imageHeightWithBorder);

	// vX is the projected coordinate of the upper-left portion of the patch;
	// it is guaranteed within the view boundary.
	// Determine the other three corners (vUR, vLL, vLR)
	_Data vUR = _Add(sh.mVX, vBasisH4);
	_Data vLL = _Add(sh.mVX, vBasisV4);
	_Data vLR = _Add(vUR, vBasisV4);
	_Data vDontCare = _SetZero();

	// Use transpose to separate the x, y, and z components.
	// JPB WIP OPT Revisit for transpose3
	_MM_TRANSPOSE4_PS(vUR, vLL, vLR, vDontCare);

	_Data vCornersX = vUR;
	_Data vCornersY = vLL;
	_Data vCornersZ = vLR;
	_Data vCornersWidthZ = _Mul(vCornersZ, sh.mVImageWidthWithBorder);
	_Data vCornersHeightZ = _Mul(vCornersZ, sh.mVImageHeightWithBorder);

	// Cmp leaves 1's if true, 0 otherwise.
	const _Data vLtX = _CmpLT(vCornersX, vCornersZ);
	const _Data vLtY = _CmpLT(vCornersY, vCornersZ);
	const _Data vGtX = _CmpGT(vCornersX, vCornersWidthZ);
	const _Data vGtY = _CmpGT(vCornersY, vCornersHeightZ);

	// Reject the patch if any of these corners are outside the view boundary.
	// JPB WIP BUG This does not identify patches which completely cover the view boundary.
	const _Data vCmpResult = _Or(_Or(vLtX, vLtY), _Or(vGtX, vGtY));

	return AllZerosI(_CastIF(vCmpResult));
}

// compute pixel's NCC score in the given target image
float DepthEstimator::ScorePixelImage(
	bool& deferred,
	_Data& deferredResult,
	const DepthData::ViewData& image1,
	const ScoreHelper& sh
)
{
	// center a patch of given size on the segment and fetch the pixel values in the target image
	float sum(0);
	#if DENSE_NCC != DENSE_NCC_DEFAULT
	float sumSq(0), num(0);
	#endif

	const Weight& w = *pWeightMap;

	const int elemByteStride = (int) image1.image.elem_stride();
	const int rowByteStride = (int) image1.image.row_stride();
	const int rowFloatStride = rowByteStride/sizeof(float);
	const _DataI vRowByteStride = _SetI(rowByteStride);
	const _DataI vRowElemStride = _SetI(elemByteStride);

	// JPB WIP BUG Hardcoded for nTexels == 25
	ASSERT(25 == nTexels);

	// JPB WIP TODO: Change scan to work by group.

	// Handle pixel "0", the ul corner, first:
	{
		// Requires accuracy.
		const _Data vPt = _Div(sh.mVX, _Splat(sh.mVX, 2));

		const _DataI vPtAsInt = _TruncateIF(vPt);
		const _Data vFracXY = _Sub(vPt, _ConvertFI(vPtAsInt));

		// Faster to calculate addresses using scalar arithmetic.
		const uchar* __restrict pSamples = image1.image.data + _AsArrayI(vPtAsInt, 1) * rowByteStride + _AsArrayI(vPtAsInt, 0) * elemByteStride;

		const float ul = *(float*) pSamples;
		const float ur = *(float*) (pSamples + elemByteStride);
		const float ll = *(float*) (pSamples + rowByteStride);
		const float lr = *(float*) (pSamples + rowByteStride + elemByteStride);

		// Lerp of lerp verified accurate with Wolfram.
		const float top = (ur - ul) * _AsArray(vFracXY, 0) + ul;
		const float bot = (lr - ll) * _AsArray(vFracXY, 0) + ll;
		const float v = (bot - top) * _AsArray(vFracXY, 1) + top;

		const float vw(v*w.pixelWeights[0]);
		sum += vw;
		sumSq += v*vw;
		num += v*w.pixelTempWeights[0];
	}

	// JPB WIP OPT Not fully AVX compatible (GROUP_SIZE must be 4).
	// DENSE_NCC == DENSE_NCC_WEIGHTED
	_Data vSampleSets[6];

	constexpr _Data vOne = {1.f, 1.f, 1.f, 1.f};

	// Do remainder of leftmost column, pixels "5", "10, "15" and "20".
	// Vectorized to process the four pixels.
	{
		const _Data xTmp = _Splat(sh.mVX, 0);
		const _Data dxTmp = _Splat(sh.mVBasisV, 0);
		constexpr _Data deltaFactor = {1.f, 2.f, 3.f, 4.f};
		_Data vX1X2X3X4 = _Add(xTmp, _Mul(dxTmp, deltaFactor));

		const _Data yTmp = _Splat(sh.mVX, 1);
		const _Data dyTmp = _Splat(sh.mVBasisV, 1);
		_Data vY1Y2Y3Y4 = _Add(yTmp, _Mul(dyTmp, deltaFactor));

		const _Data zTmp = _Splat(sh.mVX, 2);
		const _Data dzTmp = _Splat(sh.mVBasisV, 2);
		_Data vZ1Z2Z3Z4 = _Add(zTmp, _Mul(dzTmp, deltaFactor));

		// Requires accuracy.
		const _Data vPtx = _Div(vX1X2X3X4, vZ1Z2Z3Z4);
		const _Data vPty = _Div(vY1Y2Y3Y4, vZ1Z2Z3Z4);

		const _DataI vPtxAsInt = _TruncateIF(vPtx);
		const _DataI vPtyAsInt = _TruncateIF(vPty);

		// Faster to calculate addresses using scalar arithmetic.
		const uchar* __restrict pUL = image1.image.data + _AsArrayI(vPtyAsInt, 0) * rowByteStride + _AsArrayI(vPtxAsInt, 0) * elemByteStride;
		const uchar* __restrict pUR = image1.image.data + _AsArrayI(vPtyAsInt, 1) * rowByteStride + _AsArrayI(vPtxAsInt, 1) * elemByteStride;
		const uchar* __restrict pLL = image1.image.data + _AsArrayI(vPtyAsInt, 2) * rowByteStride + _AsArrayI(vPtxAsInt, 2) * elemByteStride;
		const uchar* __restrict pLR = image1.image.data + _AsArrayI(vPtyAsInt, 3) * rowByteStride + _AsArrayI(vPtxAsInt, 3) * elemByteStride;

		// Okay on rounding as all values are guaranteed positive.
		const _Data vFracX1X2X3X4 = _Sub(vPtx, _ConvertFI(vPtxAsInt)); // pt(i).x - (int) pt(i).x
		const _Data vFracY1Y2Y3Y4 = _Sub(vPty, _ConvertFI(vPtyAsInt)); // pt(i).y - (int) pt(i).y

		_Data vULSamples = _mm_loadl_pi(vY1Y2Y3Y4 /* don't care */, (__m64*) pUL);
		_Data vURSamples = _mm_loadl_pi(vY1Y2Y3Y4 /* don't care */,  (__m64*) pUR);
		_Data vLLSamples = _mm_loadl_pi(vY1Y2Y3Y4 /* don't care */, (__m64*) pLL);
		_Data vLRSamples = _mm_loadl_pi(vY1Y2Y3Y4 /* don't care */, (__m64*) pLR);

		vULSamples = _mm_loadh_pi(vULSamples, (__m64*) (pUL + rowByteStride));
		vURSamples = _mm_loadh_pi(vURSamples, (__m64*) (pUR + rowByteStride));
		vLLSamples = _mm_loadh_pi(vLLSamples, (__m64*) (pLL + rowByteStride));
		vLRSamples = _mm_loadh_pi(vLRSamples, (__m64*) (pLR + rowByteStride));

		_MM_TRANSPOSE4_PS(vULSamples, vURSamples, vLLSamples, vLRSamples);

		// Lerp of lerp verified accurate with Wolfram.
		const _Data vTop = _Add(_Mul(_Sub(vURSamples, vULSamples), vFracX1X2X3X4), vULSamples);
		const _Data vBot = _Add(_Mul(_Sub(vLRSamples, vLLSamples), vFracX1X2X3X4), vLLSamples);
		vSampleSets[0] = _Add(_Mul(_Sub(vBot, vTop), vFracY1Y2Y3Y4), vTop);
	}

	// Do the remaining 4x5 block one row at a time:
	//  1  2  3  4
	//  6  7  8  9
	// 11 12 13 14
	// 16 17 18 19
	// 21 22 23 24
	const _Data vX = _Splat(sh.mVX, 0);
	const _Data vY = _Splat(sh.mVX, 1);
	const _Data vZ = _Splat(sh.mVX, 2);

	const _Data vDx = _Splat(sh.mVBasisH, 0);
	const _Data vDy = _Splat(sh.mVBasisH, 1);
	const _Data vDz = _Splat(sh.mVBasisH, 2);

	constexpr _Data vDFactor = {1.f, 2.f, 3.f, 4.f};
	const _Data vDxDFactor = _Mul(vDx, vDFactor);
	const _Data vDyDFactor = _Mul(vDy, vDFactor);
	const _Data vDzDFactor = _Mul(vDz, vDFactor);

	_Data vX1X2X3X4 = _Add(vX, vDxDFactor);
	_Data vY1Y2Y3Y4 = _Add(vY, vDyDFactor);
	_Data vZ1Z2Z3Z4 = _Add(vZ, vDzDFactor);

	const _Data vBasisX = _Splat(sh.mVBasisV, 0);
	const _Data vBasisY = _Splat(sh.mVBasisV, 1);
	const _Data vBasisZ = _Splat(sh.mVBasisV, 2);

	for (auto i = 0; i < 5; ++i) {
		// Requires accuracy.
		// Update all coordinates for the next row.
		const _Data vPtx = _Div(vX1X2X3X4, vZ1Z2Z3Z4);
		const _Data vPty = _Div(vY1Y2Y3Y4, vZ1Z2Z3Z4);

		vX1X2X3X4 = _Add(vX1X2X3X4, vBasisX);
		vY1Y2Y3Y4 = _Add(vY1Y2Y3Y4, vBasisY);
		vZ1Z2Z3Z4 = _Add(vZ1Z2Z3Z4, vBasisZ);

		const _DataI vPtxAsInt = _TruncateIF(vPtx);
		const _DataI vPtyAsInt = _TruncateIF(vPty);

		// Faster to calculate addresses using scalar arithmetic.
		const uchar* __restrict pUL = image1.image.data + _AsArrayI(vPtyAsInt, 0) * rowByteStride + _AsArrayI(vPtxAsInt, 0) * elemByteStride;
		const uchar* __restrict pUR = image1.image.data + _AsArrayI(vPtyAsInt, 1) * rowByteStride + _AsArrayI(vPtxAsInt, 1) * elemByteStride;
		const uchar* __restrict pLL = image1.image.data + _AsArrayI(vPtyAsInt, 2) * rowByteStride + _AsArrayI(vPtxAsInt, 2) * elemByteStride;
		const uchar* __restrict pLR = image1.image.data + _AsArrayI(vPtyAsInt, 3) * rowByteStride + _AsArrayI(vPtxAsInt, 3) * elemByteStride;

		// Okay on rounding as all values are guaranteed positive.
		const _Data vFracX1X2X3X4 = _Sub(vPtx, _ConvertFI(vPtxAsInt)); // pt(i).x - (int) pt(i).x
		const _Data vFracY1Y2Y3Y4 = _Sub(vPty, _ConvertFI(vPtyAsInt)); // pt(i).y - (int) pt(i).y

		_Data vULSamples = _mm_loadl_pi(vY1Y2Y3Y4 /* don't care */, (__m64*) pUL);
		_Data vURSamples = _mm_loadl_pi(vY1Y2Y3Y4 /* don't care */,  (__m64*) pUR);
		_Data vLLSamples = _mm_loadl_pi(vY1Y2Y3Y4 /* don't care */, (__m64*) pLL);
		_Data vLRSamples = _mm_loadl_pi(vY1Y2Y3Y4 /* don't care */, (__m64*) pLR);

		vULSamples = _mm_loadh_pi(vULSamples, (__m64*) (pUL + rowByteStride));
		vURSamples = _mm_loadh_pi(vURSamples, (__m64*) (pUR + rowByteStride));
		vLLSamples = _mm_loadh_pi(vLLSamples, (__m64*) (pLL + rowByteStride));
		vLRSamples = _mm_loadh_pi(vLRSamples, (__m64*) (pLR + rowByteStride));

		_MM_TRANSPOSE4_PS(vULSamples, vURSamples, vLLSamples, vLRSamples);

		// Lerp of lerp verified accurate with Wolfram.
		const _Data vTop = _Add(_Mul(_Sub(vURSamples, vULSamples), vFracX1X2X3X4), vULSamples);
		const _Data vBot = _Add(_Mul(_Sub(vLRSamples, vLLSamples), vFracX1X2X3X4), vLLSamples);
		vSampleSets[i+1] = _Add(_Mul(_Sub(vBot, vTop), vFracY1Y2Y3Y4), vTop);
	}

	// The "4" variants will be used to maintain sums/nums for
	// four samples at a time.
	_Data vSum4 = _SetZero();
	_Data vSum4Next = _SetZero();
	_Data vSumSq4 = _SetZero();
	_Data vSumSq4Next = _SetZero();
	_Data vNum4 = _SetZero();
	_Data vNum4Next = _SetZero();

	_Data weights  = _SetN(w.pixelWeights[5], w.pixelWeights[10], w.pixelWeights[15], w.pixelWeights[20]);//_Load(&w.pixelWeights[0]); // Would have to be transposed JPB WIP BUG
	_Data tempWeights = _SetN(w.pixelTempWeights[5], w.pixelTempWeights[10], w.pixelTempWeights[15], w.pixelTempWeights[20] ); //_Load(&w.pixelTempWeights[0]);

	// Staggered on twos for better pipelining.
	for (auto i = 0; i < 6; i += 2 ) {
		if (i != 0) {
			weights = _Load(&w.pixelWeights[(i-1)*5+1]);
			tempWeights = _Load(&w.pixelTempWeights[(i-1)*5+1]);
		}

		_Data weightsNext = _Load(&w.pixelWeights[i*5+1]);
		_Data tempWeightsNext = _Load(&w.pixelTempWeights[i*5+1]);

		_Data vw = _Mul(vSampleSets[i], weights);
		_Data vwNext = _Mul(vSampleSets[i+1], weightsNext);

		vSum4 = _Add(vSum4, vw);
		vSum4Next = _Add(vSum4Next, vwNext);

		vSumSq4 = _Add(vSumSq4, _Mul(vSampleSets[i],vw));
		vSumSq4Next = _Add(vSumSq4Next, _Mul(vSampleSets[i+1],vwNext));

		vNum4 = _Add(vNum4, _Mul(vSampleSets[i], tempWeights));
		vNum4Next = _Add(vNum4Next, _Mul(vSampleSets[i+1], tempWeightsNext));
	}

	vSum4 = _Add(vSum4, vSum4Next);
	vSumSq4 = _Add(vSumSq4, vSumSq4Next);
	vNum4 = _Add(vNum4, vNum4Next);

	// Gather the single scalar sum/sumSq/num and add the 24 vectorized sum/sumSq/num
	// values together.
	sum += FastHsum(vSum4);
	sumSq += FastHsum(vSumSq4);
	num += FastHsum(vNum4);

	// score similarity of the reference and target texture patches
	#if DENSE_NCC == DENSE_NCC_FAST
	const float normSq1(sumSq-SQUARE(sum/nSizeWindow));
	#elif DENSE_NCC == DENSE_NCC_WEIGHTED
	const float normSq1(sumSq-SQUARE(sum)/w.sumWeights);
	#else
	const float normSq1(normSqDelta<float,float,nTexels>(texels1.data(), sum/(float)nTexels));
	#endif
	const float nrmSq(normSq0*normSq1);
	if (nrmSq <=1e-16f)
		return thRobust;
	#if DENSE_NCC == DENSE_NCC_DEFAULT
	const float num(texels0.dot(texels1));
	#endif
	const float ncc(FastClampS(num/FastSqrtS(nrmSq), -1.f, 1.f));
	float score(1.f-ncc);
	#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
	score *= sh.mScoreFactor;
	#endif

	if (!image1.depthMap.empty()) {
		ASSERT(OPTDENSE::fEstimationGeometricWeight > 0);
		float consistency(4.f);

		// JPB WIP BUG This should be doable with a matrix multiply with SIMD.. maybe we can keep mDepthMapPt splatted?
		// _Data tmp = _Add(_Mul(row2, depthmap2), _Add(_Mul(row0, depthmap0), _Mul(row1, depthmap1)));
		// Brings this to 

		const Point3f X1(image1.Tl*Point3f(float(X0.x)*sh.mDepth,float(X0.y)*sh.mDepth,sh.mDepth)+image1.Tm); // Kj * Rj * (Ri.t() * X + Ci - Cj)

		const float X1z = X1[2]; //image1.Tl(2,0) * sh.mDepthMapPt[0] + image1.Tl(2,1) * sh.mDepthMapPt[1] + image1.Tl(2,2) * sh.mDepthMapPt[2] + image1.Tm[2];
		if (X1z > 0.f) { // Roughly 5% rejection.
			const float X1x = X1[0]; //image1.Tl(0,0) * sh.mDepthMapPt[0] + image1.Tl(0,1) * sh.mDepthMapPt[1] + image1.Tl(0,2) * sh.mDepthMapPt[2] + image1.Tm[0];
			if (
				(X1x >= 1.f * X1z)
				& (X1x <= ((float)image1.depthMap.width() - 2)*X1z)
				) { // Roughly 1/3 are rejected,
				const float X1y = X1[1]; //image1.Tl(1,0) * sh.mDepthMapPt[0] + image1.Tl(1,1) * sh.mDepthMapPt[1] + image1.Tl(1,2) * sh.mDepthMapPt[2] + image1.Tm[1];
				if (
					(X1y >= 1.f * X1z)
					& (X1y <= ((float)image1.depthMap.height() - 2)*X1z)
					) { // 1/2 are rejected.
					const float X1PxAsFloat = X1x/X1z;
					const float X1PyAsFloat = X1y/X1z;

					const int X1PAsInt = _cvt_ftoi_fast(X1PxAsFloat);
					const int XYPAsInt = _cvt_ftoi_fast(X1PyAsFloat);

					// Depth image may differ in size.
					const int byteStride = (int) image1.depthMap.row_stride();
					const int stride = byteStride/sizeof(float);

					const size_t idx = XYPAsInt * stride + X1PAsInt;
					const float* __restrict p1 = ((float*) (image1.depthMap.data)) + idx;

					_Data vSamples = _mm_loadl_pi(vOne /* Don't care */, (__m64*) p1);
					vSamples = _mm_loadh_pi(vSamples, (__m64*) (p1+stride));

					_Data vX1Z = _Set(X1z);
					//x0y0, x1y0, x0y1, x1y1);
					constexpr _Data vSmall = {0.03f, 0.03f, 0.03f, 0.03f};
					const auto vCmp = _CmpLT(FastAbs(_Sub(vX1Z, vSamples)), _Mul(vX1Z, vSmall));

					if (!AllZerosI(_CastIF(vCmp))) { // b00 | b10 | b01 | b11) {
						const float sx = X1PxAsFloat - ((float) X1PAsInt);
						const float sy = X1PyAsFloat - ((float) XYPAsInt);
						const float sx1 = 1.f - sx;
						const float sy1 = 1.f - sy;

						//const float top = (uRSample - uLSample) * _AsArray(fracXYZ, 0) + uLSample;
						//const float bot = (lRSample - lLSample) * _AsArray(fracXYZ, 0) + uLSample;
						//const float sample = (bot - top) * _AsArray(fracXYZ, 1) + top;

						const bool b00 = _AsArrayI(_CastIF(vCmp), 0) == 0xFFFFFFFF;
						const bool b10 = _AsArrayI(_CastIF(vCmp), 1) == 0xFFFFFFFF;
						const bool b01 = _AsArrayI(_CastIF(vCmp), 2) == 0xFFFFFFFF;
						const bool b11 = _AsArrayI(_CastIF(vCmp), 3) == 0xFFFFFFFF;

						const float x0y0 = _AsArray(vSamples, 0);
						const float x1y0 = _AsArray(vSamples, 1);
						const float x0y1 = _AsArray(vSamples, 2);
						const float x1y1 = _AsArray(vSamples, 3);

						const float depth1 = 
							sy1*(sx1*Cast<float>(b00 ? x0y0 : (b10 ? x1y0 : (b01 ? x0y1 : x1y1))) + sx*Cast<float>(b10 ? x1y0 : (b00 ? x0y0 : (b11 ? x1y1 : x0y1)))) +
							sy *(sx1*Cast<float>(b01 ? x0y1 : (b11 ? x1y1 : (b00 ? x0y0 : x1y0))) + sx*Cast<float>(b11 ? x1y1 : (b01 ? x0y1 : (b10 ? x1y0 : x0y0))));

						//const Point2f xb(image1.Tr*Point3f(x1.x*depth1,x1.y*depth1,depth1)+image1.Tn); // Ki * Ri * (Rj.t() * Kj-1 * X + Cj - Ci)

						if (!sh.mLowResDepthMapEmpty) {
							const float src[] = {X1PxAsFloat * depth1, X1PyAsFloat * depth1, depth1};
							float xbx = image1.Tr(0,0) * src[0] + image1.Tr(0,1) * src[1] + image1.Tr(0,2) * src[2] + image1.Tn[0];
							float xby = image1.Tr(1,0) * src[0] + image1.Tr(1,1) * src[1] + image1.Tr(1,2) * src[2] + image1.Tn[1];
							const float xbz = image1.Tr(2,0) * src[0] + image1.Tr(2,1) * src[1] + image1.Tr(2,2) * src[2] + image1.Tn[2];

							xbx /= xbz;
							xby /= xbz;

							const float x = ((float) x0.x)-xbx;
							const float y = ((float) x0.y)-xby;
							const float dist = FastSqrtS(x*x + y*y);
							consistency = FastMinF(FastSqrtS(dist*(dist+2.f)), consistency);
						} else {
							// Defer the calculation so consistency can be calculated vectorized.
							deferredResult = image1.Tr4.Mul44Vec3(_SetN(X1PxAsFloat * depth1, X1PyAsFloat * depth1, depth1, 1.f));

							const float src[] = {X1PxAsFloat * depth1, X1PyAsFloat * depth1, depth1};
							float xbx = image1.Tr(0,0) * src[0] + image1.Tr(0,1) * src[1] + image1.Tr(0,2) * src[2] + image1.Tn[0];
							float xby = image1.Tr(1,0) * src[0] + image1.Tr(1,1) * src[1] + image1.Tr(1,2) * src[2] + image1.Tn[1];
							const float xbz = image1.Tr(2,0) * src[0] + image1.Tr(2,1) * src[1] + image1.Tr(2,2) * src[2] + image1.Tn[2];
							_AsArray(deferredResult,0) = xbx;
							_AsArray(deferredResult,1) = xby;
							_AsArray(deferredResult,2) = xbz;

							deferred = true;
							return score;
						}
					}
				}
			}
		}

		score += OPTDENSE::fEstimationGeometricWeight * consistency;
	}

	// apply depth prior weight based on patch textureless
	if (!sh.mLowResDepthMapEmpty) {
		const Depth d0 = lowResDepthMap(x0);
		if (d0 > 0) {
			const float deltaDepth(FastMinF(DepthSimilarity(d0, sh.mDepth), 0.5f));
			const float smoothSigmaDepth(-1.f / (1.f * 0.02f)); // 0.12: patch texture variance below 0.02 (0.12^2) is considered texture-less
			const float factorDeltaDepth(DENSE_EXP(normSq0 * smoothSigmaDepth));
			score = (1.f-factorDeltaDepth)*score + factorDeltaDepth*deltaDepth;
		}
	}
	ASSERT(ISFINITE(score));
	return MIN(2.f, score);
}

// compute pixel's NCC score
float DepthEstimator::ScorePixel(Depth depth, const Normal& normal, float scoreFactor)
{
	ASSERT(depth > 0 && normal.dot(Cast<float>(X0)) <= 0);
	// compute score for this pixel as seen in each view
	ASSERT(scores.size() == images.size());

	ScoreHelper sh(
		depth,
		Matx13_t(Vec3(normal).t()*INVERT(Vec3(normal).dot(X0) * depth)), // Always double.
		scoreFactor,
		Eigen::Vector4f(float(X0.x)*depth,float(X0.y)*depth,depth,1.f),
		lowResDepthMap.empty()
	);

	boost::container::small_vector<std::pair<IDX, _Data>, 1000> deferredResults;
	deferredResults.reserve(images.size());

	FOREACH(idxView, images) {
		if (IsScorable(images[idxView], sh)) {
			bool deferred = false;
			_Data deferredResult;
			scores[idxView] = ScorePixelImage(deferred, deferredResult, images[idxView], sh);
			if (deferred) { // Only sets deferred if deferred.
				deferredResults.emplace_back(idxView, deferredResult);
			}
		} else {
			scores[idxView] = thRobust;
		}
	}

	// Process the deferred elements (now more efficiently):
	if (!deferredResults.empty()) {
		const _Data vPatchX = _Set((float) x0.x);
		const _Data vPatchY = _Set((float) x0.y);

		size_t startIndex = 0;
		for (size_t remaining = deferredResults.size(); remaining > 0; startIndex += GROUP_SIZE) {
			const size_t cnt = std::min((size_t) GROUP_SIZE, remaining);
			// Beware: We may process partial groups where ultimately portions
			// of the registers may be undefined.

			_Data vConsistencies = _Set(4.f); // Default
			_Data vSrcs[4];

			for (size_t i = 0; i < cnt; ++i) {
				vSrcs[i] = deferredResults[startIndex+i].second;
			}

			_MM_TRANSPOSE4_PS(vSrcs[0], vSrcs[1], vSrcs[2], vSrcs[3]);

			// vSrcs[0] now has x0, x1, x2, x3
			auto& vx0x1x2x3 = vSrcs[0];
			auto& vy0y1y2y3 = vSrcs[1];
			auto& vz0z1z2z3 = vSrcs[2];

			_Data vProjX = _Div(vx0x1x2x3, vz0z1z2z3);
			_Data vProjY = _Div(vy0y1y2y3, vz0z1z2z3);

			_Data vDeltaX = _Sub(vPatchX, vProjX);
			_Data vDeltaY = _Sub(vPatchY, vProjY);

			_Data vDist = _Sqrt(_Add(_Mul(vDeltaX, vDeltaX), _Mul(vDeltaY, vDeltaY)));
			constexpr _Data vTwo = {2.f, 2.f, 2.f, 2.f};
			_Data vConsistenciesLhs = _Mul(vDist, _Add(vDist, vTwo));
			vConsistencies = _Min(_Sqrt(vConsistenciesLhs), vConsistencies);

			// JPB WIP OPT Revisit...
			for (size_t i = 0; i < cnt; ++i) {
				const IDX scoreIdx = deferredResults[i+startIndex].first;
				scores[scoreIdx] += OPTDENSE::fEstimationGeometricWeight * _AsArray(vConsistencies, i);

				ASSERT(ISFINITE(scores[scoreIdx]));
				scores[scoreIdx] = MIN(2.f, scores[scoreIdx]);
			}

			remaining -= cnt;
		}
	}

	#if DENSE_AGGNCC == DENSE_AGGNCC_NTH
	// set score as the nth element
	return scores.GetNth(idxScore);
	#elif DENSE_AGGNCC == DENSE_AGGNCC_MEAN
	// set score as the average similarity
	#if 1
	return scores.mean();
	#else
	const float* pscore(scores.data());
	const float* pescore(pscore+scores.rows());
	float score(0);
	do {
		score += MINF(*pscore, thRobust);
	} while (++pscore <= pescore);
	return score/scores.rows();
	#endif
	#elif DENSE_AGGNCC == DENSE_AGGNCC_MIN
	// set score as the min similarity
	return scores.minCoeff();
	#else
	// set score as the min-mean similarity
	if (idxScore == 0)
		return *std::min_element(scores.cbegin(), scores.cend());
	#if 0
	return std::accumulate(scores.cbegin(), &scores.GetNth(idxScore), 0.f) / idxScore;
	#elif 1
	float* pescore;

	// As MSVC always uses an insertion sort for GetNth when the number of items
	// is 32 or less, here we craft a much faster custom GetNth for N == 2.
	// May not work well when scores.size() is large.
	// JPB WIP Add warning.
	if (idxScore == 1 && scores.size() >= 2) {
		auto first = std::begin(scores);
		auto second = first+1;

		if (*second < *first) {
			std::iter_swap(first, second);
		}

		for (auto it = std::begin(scores)+2; it != std::end(scores); ++it) {
			if (*it < scores[0]) {
				auto tmp = scores[1];
				scores[1] = scores[0];
				scores[0] = *it;
				*it = tmp;
			} else if (*it < scores[1]) {
				std::swap(*it, scores[1]);
			}
		}
		pescore = &scores[1];
	} else {
		pescore = &scores.GetNth(idxScore);
	}
	const float* pscore(scores.cbegin());
	int n(1); float score(*pscore);
	do {
		const float s(*(++pscore));
		if (s >= thRobust)
			break;
		score += s;
		++n;
	} while (pscore < pescore);

	return score/n;
	#else
	const float thScore(MAXF(*std::min_element(scores.cbegin(), scores.cend()), 0.05f)*2);
	const float* pscore(scores.cbegin());
	const float* pescore(pscore+scores.size());
	int n(0); float score(0);
	do {
		const float s(*pscore);
		if (s <= thScore) {
			score += s;
			++n;
		}
	} while (++pscore <= pescore);
	return score/n;
	#endif
	#endif
}


float DepthEstimator::CalculateScoreFactor(_Data normal, float depth, _Data* neighborsCloseX)
{
	const _Data vSmoothBonusDepth = _Set(smoothBonusDepth);
	const _Data vSmoothSigmaNormal = _Set(smoothSigmaNormal);
	const _Data vSmoothBonusNormal = _Set(smoothBonusNormal);
	const _Data vDepthFactor = _Set(smoothSigmaDepth/(depth*depth));

	_Data neighborsPlanePosDot;
	_Data neighborsCosPlaneNormalAngle;
	const size_t numNormals = neighborsCloseNormals.size();
	for (auto i = 0; i < numNormals; ++i) {
		_AsArray(neighborsPlanePosDot, i) = FastDot4(plane, neighborsCloseX[i]);
		_AsArray(neighborsCosPlaneNormalAngle, i) = FastDot4(normal, neighborsCloseNormals[i]);
	}

	constexpr _Data vOne = {1.f, 1.f, 1.f, 1.f};
	const _Data vFdSquared = _Mul(neighborsPlanePosDot, neighborsPlanePosDot);
	const _Data vFdSquaredDepthFactor = _Mul(vFdSquared, vDepthFactor);
	const _Data vExpFdSquaredDepthFactor = FastExpAlwaysNegative(vFdSquaredDepthFactor);
	const _Data vExpFdSquaredDepthFactorBD = _Mul(vExpFdSquaredDepthFactor, vSmoothBonusDepth);
	const _Data vScoreFactorLhs = _Sub(vOne, vExpFdSquaredDepthFactorBD);

	constexpr _Data vNegOne = {-1.f, -1.f, -1.f, -1.f};
	const _Data vAnglesClamped = FastClamp(neighborsCosPlaneNormalAngle, vNegOne, vOne);
	const _Data vTmp = FastACos(vAnglesClamped);
	const _Data vTmpSquared = _Mul(vTmp, vTmp);
	const _Data vTmpSquaredvSmoothSigmaNormal = _Mul(vTmpSquared, vSmoothSigmaNormal);
	const _Data vExpTmpSquaredvSmoothSigmaNormal = FastExpAlwaysPositive(vTmpSquaredvSmoothSigmaNormal);
	const _Data vExpTmpSquaredvSmoothSigmaNormalBN = _Mul(vExpTmpSquaredvSmoothSigmaNormal, vSmoothBonusNormal);
	const _Data vScoreFactorRhs = _Sub(vOne, vExpTmpSquaredvSmoothSigmaNormalBN);

	_Data vScoreFactor = _Mul(vScoreFactorLhs, vScoreFactorRhs);
	// First (only available member to initialize) is m128i_i8
	constexpr _DataI vMasks[] = {
		{0x00,0x00,0x00,0x00,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127},
		{0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,-127,-127,-127,-127,-127,-127,-127,-127},
		{0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,-127,-127,-127,-127},
		{0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}
	};
	vScoreFactor = Blend(vScoreFactor, vOne, _CastFI(vMasks[numNormals-1]));

	return FastHProduct(vScoreFactor);
}


// run propagation and random refinement cycles;
// the solution belonging to the target image can be also propagated
void DepthEstimator::ProcessPixel(IDX idx)
{
	// compute pixel coordinates from pixel index and its neighbors
	ASSERT(dir == LT2RB || dir == RB2LT);
	if (!PreparePixelPatch(dir == LT2RB ? coords[idx] : coords[coords.GetSize()-1-idx]) || !FillPixelPatch())
		return;
	// find neighbors
	neighbors.Empty();
	#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
	neighborsCloseNormals.clear();
		#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
		boost::container::small_vector<_Data,4> neighborsCloseX;
		#endif
	boost::container::small_vector<Depth,4> neighborsCloseDepths;
	#endif
	if (dir == LT2RB) {
		// direction from left-top to right-bottom corner
		if (x0.x > nSizeHalfWindow) {
			const ImageRef nx(x0.x-1, x0.y);
			const Depth ndepth(depthMap0(nx));
			if (ndepth > 0) {
				#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
				ASSERT(ISEQUAL(norm(normalMap0(nx)), 1.f));
				neighbors.emplace_back(nx);
				const auto& nm = normalMap0(nx);
				neighborsCloseNormals.emplace_back(_SetN(nm[0], nm[1], nm[2], 0.f));
					#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
					const auto X = Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth)));
					neighborsCloseX.emplace_back(_SetN(X[0], X[1], X[2], 1.f));
					#endif
				neighborsCloseDepths.push_back(ndepth);
				#else
				neighbors.emplace_back(NeighborData{nx,ndepth,normalMap0(nx)});
				#endif
			}
		}
		if (x0.y > nSizeHalfWindow) {
			const ImageRef nx(x0.x, x0.y-1);
			const Depth ndepth(depthMap0(nx));
			if (ndepth > 0) {
				#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
				ASSERT(ISEQUAL(norm(normalMap0(nx)), 1.f));
				neighbors.emplace_back(nx);
				const auto& nm = normalMap0(nx);
				neighborsCloseNormals.emplace_back(_SetN(nm[0], nm[1], nm[2], 0.f));
					#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
					const auto X = Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth)));
					neighborsCloseX.emplace_back(_SetN(X[0], X[1], X[2], 1.f));
					#endif
				neighborsCloseDepths.push_back(ndepth);
				#else
				neighbors.emplace_back(NeighborData{nx,ndepth,normalMap0(nx)});
				#endif
			}
		}
		#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
		if (x0.x < size.width-nSizeHalfWindow) {
			const ImageRef nx(x0.x+1, x0.y);
			const Depth ndepth(depthMap0(nx));
			if (ndepth > 0) {
				ASSERT(ISEQUAL(norm(normalMap0(nx)), 1.f));
				const auto& nm = normalMap0(nx);
				neighborsCloseNormals.emplace_back(_SetN(nm[0], nm[1], nm[2], 0.f));
					#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
					const auto X = Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth)));
					neighborsCloseX.emplace_back(_SetN(X[0], X[1], X[2], 1.f));
					#endif
				neighborsCloseDepths.push_back(ndepth);
			}
		}
		if (x0.y < size.height-nSizeHalfWindow) {
			const ImageRef nx(x0.x, x0.y+1);
			const Depth ndepth(depthMap0(nx));
			if (ndepth > 0) {
				ASSERT(ISEQUAL(norm(normalMap0(nx)), 1.f));
				const auto& nm = normalMap0(nx);
				neighborsCloseNormals.emplace_back(_SetN(nm[0], nm[1], nm[2], 0.f));
					#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
					const auto X = Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth)));
					neighborsCloseX.emplace_back(_SetN(X[0], X[1], X[2], 1.f));
					#endif
				neighborsCloseDepths.push_back(ndepth);
			}
		}
		#endif
	} else {
		ASSERT(dir == RB2LT);
		// direction from right-bottom to left-top corner
		if (x0.x < size.width-nSizeHalfWindow) {
			const ImageRef nx(x0.x+1, x0.y);
			const Depth ndepth(depthMap0(nx));
			if (ndepth > 0) {
				#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
				ASSERT(ISEQUAL(norm(normalMap0(nx)), 1.f));
				neighbors.emplace_back(nx);
				const auto& nm = normalMap0(nx);
				neighborsCloseNormals.emplace_back(_SetN(nm[0], nm[1], nm[2], 0.f));
					#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
					const auto X = Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth)));
					neighborsCloseX.emplace_back(_SetN(X[0], X[1], X[2], 1.f));
					#endif
				neighborsCloseDepths.push_back(ndepth);
				#else
				neighbors.emplace_back(NeighborData{nx,ndepth,normalMap0(nx)});
				#endif
			}
		}
		if (x0.y < size.height-nSizeHalfWindow) {
			const ImageRef nx(x0.x, x0.y+1);
			const Depth ndepth(depthMap0(nx));
			if (ndepth > 0) {
				#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
				ASSERT(ISEQUAL(norm(normalMap0(nx)), 1.f));
				neighbors.emplace_back(nx);
				const auto& nm = normalMap0(nx);
				neighborsCloseNormals.emplace_back(_SetN(nm[0], nm[1], nm[2], 0.f));
					#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
					const auto X = Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth)));
					neighborsCloseX.emplace_back(_SetN(X[0], X[1], X[2], 1.f));
					#endif
				neighborsCloseDepths.push_back(ndepth);
				#else
				neighbors.emplace_back(NeighborData{nx,ndepth,normalMap0(nx)});
				#endif
			}
		}
		#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
		if (x0.x > nSizeHalfWindow) {
			const ImageRef nx(x0.x-1, x0.y);
			const Depth ndepth(depthMap0(nx));
			if (ndepth > 0) {
				ASSERT(ISEQUAL(norm(normalMap0(nx)), 1.f));
				const auto& nm = normalMap0(nx);
				neighborsCloseNormals.emplace_back(_SetN(nm[0], nm[1], nm[2], 0.f));
					#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
					const auto X = Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth)));
					neighborsCloseX.emplace_back(_SetN(X[0], X[1], X[2], 1.f));
					#endif
				neighborsCloseDepths.push_back(ndepth);
			}
		}
		if (x0.y > nSizeHalfWindow) {
			const ImageRef nx(x0.x, x0.y-1);
			const Depth ndepth(depthMap0(nx));
			if (ndepth > 0) {
				ASSERT(ISEQUAL(norm(normalMap0(nx)), 1.f));
				const auto& nm = normalMap0(nx);
				neighborsCloseNormals.emplace_back(_SetN(nm[0], nm[1], nm[2], 0.f));
					#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
					const auto X = Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth)));
					neighborsCloseX.emplace_back(_SetN(X[0], X[1], X[2], 1.f));
					#endif
				neighborsCloseDepths.push_back(ndepth);
			}
		}
		#endif
	}
	float& conf = confMap0(x0);
	Depth& depth = depthMap0(x0);
	Normal& normal = normalMap0(x0);
	const Normal viewDir(Cast<float>(X0));
	ASSERT(depth > 0 && normal.dot(viewDir) <= 0);
	#if DENSE_REFINE == DENSE_REFINE_ITER
	// check if any of the neighbor estimates are better then the current estimate
	#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
	FOREACH(n, neighbors) {
		const ImageRef& nx = neighbors[n];
	#else
	for (NeighborData& neighbor: neighbors) {
		const ImageRef& nx = neighbor.x;
	#endif
		if (confMap0(nx) >= OPTDENSE::fNCCThresholdKeep)
			continue;
		// I believe only the #if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA works properly.
		#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
		_Data neighborNormal = neighborsCloseNormals[n];
		#endif
		Normal nn = Normal(_AsArray(neighborNormal, 0), _AsArray(neighborNormal, 1), _AsArray(neighborNormal, 2));
		Depth neighborDepth = InterpolatePixel(nx, neighborsCloseDepths[n], nn);
		CorrectNormal(nn);
		//ASSERT(neighbor.depth > 0 && neighbor.normal.dot(viewDir) <= 0);
		float scoreFactor = 1.f; // Default
		#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
		if (!neighborsCloseNormals.empty()) {
			InitPlane(neighborDepth, nn);
			scoreFactor = CalculateScoreFactor(neighborNormal, neighborDepth, neighborsCloseX.data());
		}
		#endif
		const float nconf(ScorePixel(neighborDepth, nn, scoreFactor));
		ASSERT(nconf >= 0 && nconf <= 2);
		if (conf > nconf) {
			conf = nconf;
			depth = neighborDepth;
			normal = Normal(_AsArray(neighborNormal, 0), _AsArray(neighborNormal, 1), _AsArray(neighborNormal, 2));
		}
	}

	// try random values around the current estimate in order to refine it
	unsigned idxScaleRange(0);
	RefineIters:
	if (conf <= thConfSmall)
		idxScaleRange = 2;
	else if (conf <= thConfBig)
		idxScaleRange = 1;
	else if (conf >= thConfRand) {
		// try completely random values in order to find an initial estimate
		#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
		neighborsCloseNormals.clear();
		#endif
		for (unsigned iter=0; iter<OPTDENSE::nRandomIters; ++iter) {
			const Depth ndepth(RandomDepth(dMinSqr, dMaxSqr));
			const Normal nnormal(RandomNormal(viewDir));
			const float nconf(ScorePixel(ndepth, nnormal));
			ASSERT(nconf >= 0);
			if (conf > nconf) {
				conf = nconf;
				depth = ndepth;
				normal = nnormal;
				if (conf < thConfRand)
					goto RefineIters;
			}
		}
		return;
	}
	float scaleRange(scaleRanges[idxScaleRange]);
	const float depthRange(MaxDepthDifference(depth, OPTDENSE::fRandomDepthRatio));
	Point2f p;
	Normal2Dir(normal, p);
	Normal nnormal;
	for (unsigned iter=0; iter<OPTDENSE::nRandomIters; ++iter) {
		const Depth ndepth(rnd.randomMeanRange(depth, depthRange*scaleRange));
		if (!ISINSIDE(ndepth, dMin, dMax))
			continue;
		const Point2f np(rnd.randomMeanRange(p.x, angle1Range*scaleRange), rnd.randomMeanRange(p.y, angle2Range*scaleRange));
		Dir2Normal(np, nnormal);
		if (nnormal.dot(viewDir) >= 0)
			continue;
		float scoreFactor = 1.f;
		#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
		if (!neighborsCloseNormals.empty()) {
			InitPlane(ndepth, nnormal);
			scoreFactor = CalculateScoreFactor(_SetN(nnormal[0], nnormal[1], nnormal[2], 0.f), ndepth, neighborsCloseX.data());
		}
		#endif
		const float nconf(ScorePixel(ndepth, nnormal, scoreFactor));
		ASSERT(nconf >= 0);
		if (conf > nconf) {
			conf = nconf;
			depth = ndepth;
			normal = nnormal;
			p = np;
			scaleRange = scaleRanges[++idxScaleRange];
		}
	}
	#else
	// current pixel estimate
	PixelEstimate currEstimate{depth, normal};
	// propagate depth estimate from the best neighbor estimate
	PixelEstimate prevEstimate; float prevCost(FLT_MAX);
	#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
	FOREACH(n, neighbors) {
		const ImageRef& nx = neighbors[n];
	#else
	for (const NeighborData& neighbor: neighbors) {
		const ImageRef& nx = neighbor.x;
	#endif
		float nconf(confMap0(nx));
		const unsigned nidxScaleRange(DecodeScoreScale(nconf));
		ASSERT(nconf >= 0 && nconf <= 2);
		if (nconf >= OPTDENSE::fNCCThresholdKeep)
			continue;
		if (prevCost <= nconf)
			continue;
		#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
		const NeighborEstimate& neighbor = neighborsClose[n];
		#endif
		if (neighbor.normal.dot(viewDir) >= 0)
			continue;
		prevEstimate.depth = InterpolatePixel(nx, neighbor.depth, neighbor.normal);
		prevEstimate.normal = neighbor.normal;
		CorrectNormal(prevEstimate.normal);
		prevCost = nconf;
	}
	if (prevCost == FLT_MAX)
		prevEstimate = PerturbEstimate(currEstimate, thPerturbation);
	// randomly sampled estimate
	PixelEstimate randEstimate(PerturbEstimate(currEstimate, thPerturbation));
	// select best pixel estimate
	const int numCosts = 5;
	float costs[numCosts] = {0,0,0,0,0};
	const Depth depths[numCosts] = {
		currEstimate.depth, prevEstimate.depth, randEstimate.depth,
		currEstimate.depth, randEstimate.depth};
	const Normal normals[numCosts] = {
		currEstimate.normal, prevEstimate.normal,
		randEstimate.normal, randEstimate.normal,
		currEstimate.normal};
	conf = FLT_MAX;
	for (int idxCost=0; idxCost<numCosts; ++idxCost) {
		const Depth ndepth(depths[idxCost]);
		const Normal nnormal(normals[idxCost]);
		#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
		InitPlane(ndepth, nnormal);
		#endif
		const float nconf(ScorePixel(ndepth, nnormal));
		ASSERT(nconf >= 0);
		if (conf > nconf) {
			conf = nconf;
			depth = ndepth;
			normal = nnormal;
		}
	}
	#endif
}

// interpolate given pixel's estimate to the current position
Depth DepthEstimator::InterpolatePixel(const ImageRef& nx, Depth depth, const Normal& normal) const
{
	ASSERT(depth > 0 && normal.dot(image0.camera.TransformPointI2C(Cast<REAL>(nx))) <= 0);
	Depth depthNew;
	#if 1
	// compute as intersection of the lines
	// {(x1, y1), (x2, y2)} from neighbor's 3D point towards normal direction
	// and
	// {(0, 0), (x4, 1)} from camera center towards current pixel direction
	// in the x or y plane
	if (x0.x == nx.x) {
		const float nx1((float)(((REAL)x0.y - image0.camera.K(1,2)) / image0.camera.K(1,1)));
		const float denom(normal.z + nx1 * normal.y);
		if (ISZERO(denom))
			return depth;
		const float x1((float)(((REAL)nx.y - image0.camera.K(1,2)) / image0.camera.K(1,1)));
		const float nom(depth * (normal.z + x1 * normal.y));
		depthNew = nom / denom;
	}
	else {
		ASSERT(x0.y == nx.y);
		const float nx1((float)(((REAL)x0.x - image0.camera.K(0,2)) / image0.camera.K(0,0)));
		const float denom(normal.z + nx1 * normal.x);
		if (ISZERO(denom))
			return depth;
		const float x1((float)(((REAL)nx.x - image0.camera.K(0,2)) / image0.camera.K(0,0)));
		const float nom(depth * (normal.z + x1 * normal.x));
		depthNew = nom / denom;
	}
	#else
	// compute as the ray - plane intersection
	{
		#if 0
		const Plane plane(Cast<REAL>(normal), image0.camera.TransformPointI2C(Point3(nx, depth)));
		const Ray3 ray(Point3::ZERO, normalized(X0));
		depthNew = (Depth)ray.Intersects(plane).z();
		#else
		const Point3 planeN(normal);
		const REAL planeD(planeN.dot(image0.camera.TransformPointI2C(Point3(nx, depth))));
		depthNew = (Depth)(planeD / planeN.dot(X0));
		#endif
	}
	#endif
	return ISINSIDE(depthNew,dMin,dMax) ? depthNew : depth;
}

#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
// compute plane defined by current depth and normal estimate
void DepthEstimator::InitPlane(Depth depth, const Normal& normal)
{
#if 0
	plane.Set(normal, Normal(Cast<float>(X0)*depth));
#else
	plane = _SetN(normal[0], normal[1], normal[2], -depth*normal.dot(Cast<float>(X0)));
#endif
}
#endif

#if DENSE_REFINE == DENSE_REFINE_EXACT
DepthEstimator::PixelEstimate DepthEstimator::PerturbEstimate(const PixelEstimate& est, float perturbation)
{
	PixelEstimate ptbEst;

	// perturb depth
	const float minDepth = est.depth * (1.f-perturbation);
	const float maxDepth = est.depth * (1.f+perturbation);
	ptbEst.depth = CLAMP(rnd.randomUniform(minDepth, maxDepth), dMin, dMax);

	// perturb normal
	const Normal viewDir(Cast<float>(X0));
	std::uniform_real_distribution<float> urd(-1.f, 1.f);
	const int numMaxTrials = 3;
	int numTrials = 0;
	perturbation *= FHALF_PI;
	while(true) {
		// generate random perturbation rotation
		const RMatrixBaseF R(urd(rnd)*perturbation, urd(rnd)*perturbation, urd(rnd)*perturbation);
		// perturb normal vector
		ptbEst.normal = R * est.normal;
		// make sure the perturbed normal is still looking towards the camera,
		// otherwise try again with a smaller perturbation
		if (ptbEst.normal.dot(viewDir) < 0.f)
			break;
		if (++numTrials == numMaxTrials) {
			ptbEst.normal = est.normal;
			return ptbEst;
		}
		perturbation *= 0.5f;
	}
	ASSERT(ISEQUAL(norm(ptbEst.normal), 1.f));

	return ptbEst;
}
#endif
/*----------------------------------------------------------------*/



// S T R U C T S ///////////////////////////////////////////////////

namespace CGAL {
}

// triangulate in-view points, generating a 2D mesh
// return also the estimated depth boundaries (min and max depth)
std::pair<float,float> TriangulatePointsDelaunay(const DepthData::ViewData& image, const PointCloud& pointcloud, const IndexArr& points, Mesh& mesh, Point2fArr& projs, bool bAddCorners)
{
	typedef CGAL::Simple_cartesian<double> kernel_t;
	typedef CGAL::Triangulation_vertex_base_with_info_2<Mesh::VIndex, kernel_t> vertex_base_t;
	typedef CGAL::Triangulation_data_structure_2<vertex_base_t> triangulation_data_structure_t;
	typedef CGAL::Delaunay_triangulation_2<kernel_t, triangulation_data_structure_t> Delaunay;
	typedef Delaunay::Face_circulator FaceCirculator;
	typedef Delaunay::Face_handle FaceHandle;
	typedef Delaunay::Vertex_handle VertexHandle;
	typedef kernel_t::Point_2 CPoint;

	ASSERT(sizeof(Point3) == sizeof(X3D));
	ASSERT(sizeof(Point2) == sizeof(CPoint));
	std::pair<float,float> depthBounds(FLT_MAX, 0.f);
	mesh.vertices.reserve((Mesh::VIndex)points.size()+4);
	projs.reserve(mesh.vertices.capacity());
	Delaunay delaunay;
	for (uint32_t idx: points) {
		const Point3f pt(image.camera.ProjectPointP3(pointcloud.points[idx]));
		const Point3f x(pt.x/pt.z, pt.y/pt.z, pt.z);
		delaunay.insert(CPoint(x.x, x.y))->info() = mesh.vertices.size();
		mesh.vertices.emplace_back(image.camera.TransformPointI2C(x));
		projs.emplace_back(x.x, x.y);
		if (depthBounds.first > pt.z)
			depthBounds.first = pt.z;
		if (depthBounds.second < pt.z)
			depthBounds.second = pt.z;
	}
	// if full size depth-map requested
	const size_t numPoints(3);
	if (bAddCorners && points.size() >= numPoints) {
		// add the four image corners at the average depth
		ASSERT(image.pImageData->IsValid() && ISINSIDE(image.pImageData->avgDepth, depthBounds.first, depthBounds.second));
		const Mesh::VIndex idxFirstVertex = mesh.vertices.size();
		VertexHandle vcorners[4];
		for (const Point2f x: {Point2i(0, 0), Point2i(image.image.width()-1, 0), Point2i(0, image.image.height()-1), Point2i(image.image.width()-1, image.image.height()-1)}) {
			const Mesh::VIndex i(mesh.vertices.size() - idxFirstVertex);
			(vcorners[i] = delaunay.insert(CPoint(x.x, x.y)))->info() = mesh.vertices.size();
			mesh.vertices.emplace_back(image.camera.TransformPointI2C(Point3f(x, image.pImageData->avgDepth)));
			projs.emplace_back(x);
		}
		// compute average depth from the closest 3 directly connected faces,
		// weighted by the distance
		for (int i=0; i<4; ++i) {
			const VertexHandle vcorner(vcorners[i]);
			FaceCirculator cfc(delaunay.incident_faces(vcorner));
			ASSERT(cfc != 0);
			const FaceCirculator done(cfc);
			const Point2d& posA = reinterpret_cast<const Point2d&>(vcorner->point());
			const Ray3d rayA(Point3d::ZERO, normalized(Cast<REAL>(mesh.vertices[vcorner->info()])));
			typedef TIndexScore<float,float> DepthDist;
			CLISTDEF0(DepthDist) depths(0, numPoints);
			do {
				const FaceHandle fc(cfc->neighbor(cfc->index(vcorner)));
				if (delaunay.is_infinite(fc))
					continue;
				for (int j=0; j<4; ++j)
					if (fc->has_vertex(vcorners[j]))
						continue;
				// compute the depth as the intersection of the corner ray with
				// the plane defined by the face's vertices
				const Planed planeB(
					Cast<REAL>(mesh.vertices[fc->vertex(0)->info()]),
					Cast<REAL>(mesh.vertices[fc->vertex(1)->info()]),
					Cast<REAL>(mesh.vertices[fc->vertex(2)->info()])
				);
				const Point3d poszB(rayA.Intersects(planeB));
				if (poszB.z <= 0)
					continue;
				const Point2d posB((
					reinterpret_cast<const Point2d&>(fc->vertex(0)->point())+
					reinterpret_cast<const Point2d&>(fc->vertex(1)->point())+
					reinterpret_cast<const Point2d&>(fc->vertex(2)->point()))/3.f
				);
				const double dist(norm(posB-posA));
				depths.StoreTop<numPoints>(DepthDist(CLAMP((float)poszB.z,depthBounds.first,depthBounds.second), INVERT((float)dist)));
			} while (++cfc != done);
			ASSERT(depths.size() == numPoints);
			typedef Eigen::Map< Eigen::VectorXf, Eigen::Unaligned, Eigen::InnerStride<2> > FloatMap;
			FloatMap vecDists(&depths[0].score, numPoints);
			vecDists *= 1.f/vecDists.sum();
			FloatMap vecDepths(&depths[0].idx, numPoints);
			const float depth(vecDepths.dot(vecDists));
			mesh.vertices[idxFirstVertex+i] = image.camera.TransformPointI2C(Point3(posA, depth));
		}
	}
	mesh.faces.reserve(Mesh::FIndex(std::distance(delaunay.finite_faces_begin(),delaunay.finite_faces_end())));
	for (Delaunay::Face_iterator it=delaunay.faces_begin(); it!=delaunay.faces_end(); ++it) {
		const Delaunay::Face& face = *it;
		mesh.faces.emplace_back(face.vertex(2)->info(), face.vertex(1)->info(), face.vertex(0)->info());
	}
	return depthBounds;
}

// roughly estimate depth and normal maps by triangulating the sparse point cloud
// and interpolating normal and depth for all pixels
bool MVS::TriangulatePoints2DepthMap(
	const DepthData::ViewData& image, const PointCloud& pointcloud, const IndexArr& points,
	DepthMap& depthMap, NormalMap& normalMap, Depth& dMin, Depth& dMax, bool bAddCorners, bool bSparseOnly)
{
	ASSERT(image.pImageData != NULL);

	// triangulate in-view points
	Mesh mesh;
	Point2fArr projs;
	const std::pair<float,float> thDepth(TriangulatePointsDelaunay(image, pointcloud, points, mesh, projs, bAddCorners));
	dMin = thDepth.first;
	dMax = thDepth.second;

	// create rough depth-map by interpolating inside triangles
	const Camera& camera = image.camera;
	mesh.ComputeNormalVertices();
	depthMap.create(image.image.size());
	normalMap.create(image.image.size());
	if (!bAddCorners || bSparseOnly) {
		depthMap.memset(0);
		normalMap.memset(0);
	}
	if (bSparseOnly) {
		// just project sparse pointcloud onto depthmap
		FOREACH(i, mesh.vertices) {
			const Point2f& x(projs[i]);
			const Point2i ix(FLOOR2INT(x));
			const Depth z(mesh.vertices[i].z);
			const Normal& normal(mesh.vertexNormals[i]);
			for (const Point2i dx : {Point2i(0,0),Point2i(1,0),Point2i(0,1),Point2i(1,1)}) {
				const Point2i ax(ix + dx);
				if (!depthMap.isInside(ax))
					continue;
				depthMap(ax) = z;
				normalMap(ax) = normal;
			}
		}
	} else {
		// rasterize triangles onto depthmap
		struct RasterDepth : TRasterMeshBase<RasterDepth> {
			typedef TRasterMeshBase<RasterDepth> Base;
			using Base::camera;
			using Base::depthMap;
			using Base::ptc;
			using Base::pti;
			const Mesh::NormalArr& vertexNormals;
			NormalMap& normalMap;
			Mesh::Face face;
			RasterDepth(const Mesh::NormalArr& _vertexNormals, const Camera& _camera, DepthMap& _depthMap, NormalMap& _normalMap)
				: Base(_camera, _depthMap), vertexNormals(_vertexNormals), normalMap(_normalMap) {}
			inline void operator()(const ImageRef& pt, const Point3f& bary) {
				const Point3f pbary(PerspectiveCorrectBarycentricCoordinates(bary));
				const Depth z(ComputeDepth(pbary));
				ASSERT(z > Depth(0));
				depthMap(pt) = z;
				normalMap(pt) = normalized(
					vertexNormals[face[0]] * pbary[0]+
					vertexNormals[face[1]] * pbary[1]+
					vertexNormals[face[2]] * pbary[2]
				);
			}
		};
		RasterDepth rasterer = {mesh.vertexNormals, camera, depthMap, normalMap};
		for (const Mesh::Face& face : mesh.faces) {
			rasterer.face = face;
			rasterer.ptc[0].z = mesh.vertices[face[0]].z;
			rasterer.ptc[1].z = mesh.vertices[face[1]].z;
			rasterer.ptc[2].z = mesh.vertices[face[2]].z;
			Image8U::RasterizeTriangleBary(
				projs[face[0]],
				projs[face[1]],
				projs[face[2]], rasterer);
		}
	}
	return true;
} // TriangulatePoints2DepthMap
// same as above, but does not estimate the normal-map
bool MVS::TriangulatePoints2DepthMap(
	const DepthData::ViewData& image, const PointCloud& pointcloud, const IndexArr& points,
	DepthMap& depthMap, Depth& dMin, Depth& dMax, bool bAddCorners, bool bSparseOnly)
{
	ASSERT(image.pImageData != NULL);

	// triangulate in-view points
	Mesh mesh;
	Point2fArr projs;
	const std::pair<float,float> thDepth(TriangulatePointsDelaunay(image, pointcloud, points, mesh, projs, bAddCorners));
	dMin = thDepth.first;
	dMax = thDepth.second;

	// create rough depth-map by interpolating inside triangles
	const Camera& camera = image.camera;
	depthMap.create(image.image.size());
	if (!bAddCorners || bSparseOnly)
		depthMap.memset(0);
	if (bSparseOnly) {
		// just project sparse pointcloud onto depthmap
		FOREACH(i, mesh.vertices) {
			const Point2f& x(projs[i]);
			const Point2i ix(FLOOR2INT(x));
			const Depth z(mesh.vertices[i].z);
			for (const Point2i dx : {Point2i(0,0),Point2i(1,0),Point2i(0,1),Point2i(1,1)}) {
				const Point2i ax(ix + dx);
				if (!depthMap.isInside(ax))
					continue;
				depthMap(ax) = z;
			}
		}
	} else {
		// rasterize triangles onto depthmap
		struct RasterDepth : TRasterMeshBase<RasterDepth> {
			typedef TRasterMeshBase<RasterDepth> Base;
			using Base::depthMap;
			RasterDepth(const Camera& _camera, DepthMap& _depthMap)
				: Base(_camera, _depthMap) {}
			inline void operator()(const ImageRef& pt, const Point3f& bary) {
				const Point3f pbary(PerspectiveCorrectBarycentricCoordinates(bary));
				const Depth z(ComputeDepth(pbary));
				ASSERT(z > Depth(0));
				depthMap(pt) = z;
			}
		};
		RasterDepth rasterer = {camera, depthMap};
		for (const Mesh::Face& face : mesh.faces) {
			rasterer.ptc[0].z = mesh.vertices[face[0]].z;
			rasterer.ptc[1].z = mesh.vertices[face[1]].z;
			rasterer.ptc[2].z = mesh.vertices[face[2]].z;
			Image8U::RasterizeTriangleBary(
				projs[face[0]],
				projs[face[1]],
				projs[face[2]], rasterer);
		}
	}
	return true;
} // TriangulatePoints2DepthMap
/*----------------------------------------------------------------*/


namespace MVS {

// least squares refinement of the given plane to the 3D point set
// (return the number of iterations)
template <typename TYPE>
int OptimizePlane(TPlane<TYPE,3>& plane, const Eigen::Matrix<TYPE,3,1>* points, size_t size, int maxIters, TYPE threshold)
{
	typedef TPlane<TYPE,3> PLANE;
	typedef Eigen::Matrix<TYPE,3,1> POINT;
	ASSERT(size >= PLANE::numParams);
	struct OptimizationFunctor {
		const POINT* points;
		const size_t size;
		const RobustNorm::GemanMcClure<double> robust;
		// construct with the data points
		OptimizationFunctor(const POINT* _points, size_t _size, double _th)
			: points(_points), size(_size), robust(_th) { ASSERT(size < (size_t)std::numeric_limits<int>::max()); }
		static void Residuals(const double* x, int nPoints, const void* pData, double* fvec, double* fjac, int* /*info*/) {
			const OptimizationFunctor& data = *reinterpret_cast<const OptimizationFunctor*>(pData);
			ASSERT((size_t)nPoints == data.size && fvec != NULL && fjac == NULL);
			TPlane<double,3> plane; {
				Point3d N;
				plane.m_fD = x[0];
				Dir2Normal(reinterpret_cast<const Point2d&>(x[1]), N);
				plane.m_vN = N;
			}
			for (size_t i=0; i<data.size; ++i)
				fvec[i] = data.robust(plane.Distance(data.points[i].template cast<double>()));
		}
	} functor(points, size, threshold);
	double arrParams[PLANE::numParams]; {
		arrParams[0] = (double)plane.m_fD;
		const Point3d N(plane.m_vN.x(), plane.m_vN.y(), plane.m_vN.z());
		Normal2Dir(N, reinterpret_cast<Point2d&>(arrParams[1]));
	}
	lm_control_struct control = {1.e-6, 1.e-7, 1.e-8, 1.e-7, 100.0, maxIters}; // lm_control_float;
	lm_status_struct status;
	lmmin(PLANE::numParams, arrParams, (int)size, &functor, OptimizationFunctor::Residuals, &control, &status);
	switch (status.info) {
	//case 4:
	case 5:
	case 6:
	case 7:
	case 8:
	case 9:
	case 10:
	case 11:
	case 12:
		DEBUG_ULTIMATE("error: refine plane: %s", lm_infmsg[status.info]);
		return 0;
	}
	{
		Point3d N;
		plane.m_fD = (TYPE)arrParams[0];
		Dir2Normal(reinterpret_cast<const Point2d&>(arrParams[1]), N);
		plane.m_vN = Cast<TYPE>(N);
	}
	return status.nfev;
}

template <typename TYPE>
class TPlaneSolverAdaptor
{
public:
	enum { MINIMUM_SAMPLES = 3 };
	enum { MAX_MODELS = 1 };

	typedef TYPE Type;
	typedef TPoint3<TYPE> Point;
	typedef CLISTDEF0(Point) Points;
	typedef TPlane<TYPE,3> Model;
	typedef CLISTDEF0(Model) Models;

	TPlaneSolverAdaptor(const Points& points)
		: points_(points)
	{
	}
	TPlaneSolverAdaptor(const Points& points, float w, float h, float d)
		: points_(points)
	{
		// LogAlpha0 is used to make error data scale invariant
		// Ratio of containing diagonal image rectangle over image area
		const float D = SQRT(w*w + h*h + d*d); // diameter
		const float A = w*h*d+1.f; // volume
		logalpha0_ = LOG10(2.f*D/A*0.5f);
	}

	inline bool Fit(const std::vector<size_t>& samples, Models& models) const {
		ASSERT(samples.size() == MINIMUM_SAMPLES);
		Point points[MINIMUM_SAMPLES];
		for (size_t i=0; i<MINIMUM_SAMPLES; ++i)
			points[i] = points_[samples[i]];
		if (CheckCollinearity(points, 3))
			return false;
		models.Resize(1);
		models[0] = Model(points[0], points[1], points[2]);
		return true;
	}

	inline void EvaluateModel(const Model& model) {
		model2evaluate = model;
	}

	inline double Error(size_t sample) const {
		return SQUARE(model2evaluate.Distance(points_[sample]));
	}

	static double Error(const Model& plane, const Points& points) {
		double e(0);
		for (const Point& X: points)
			e += plane.DistanceAbs(X);
		return e/points.size();
	}

	inline size_t NumSamples() const { return static_cast<size_t>(points_.size()); }
	inline double logalpha0() const { return logalpha0_; }
	inline double multError() const { return 0.5; }

protected:
	const Points& points_; // Normalized input data
	double logalpha0_; // Alpha0 is used to make the error adaptive to the image size
	Model model2evaluate; // current model to be evaluated
};

// Robustly estimate the plane that fits best the given points
template <typename TYPE, typename Sampler, bool bFixThreshold>
unsigned TEstimatePlane(const CLISTDEF0(TPoint3<TYPE>)& points, TPlane<TYPE,3>& plane, double& maxThreshold, bool arrInliers[], size_t maxIters)
{
	typedef TPlaneSolverAdaptor<TYPE> PlaneSolverAdaptor;

	plane.Invalidate();
	
	const unsigned nPoints = (unsigned)points.size();
	if (nPoints < PlaneSolverAdaptor::MINIMUM_SAMPLES) {
		ASSERT("too few points" == NULL);
		return 0;
	}

	// normalize points
	TMatrix<TYPE,4,4> H;
	typename PlaneSolverAdaptor::Points normPoints;
	NormalizePoints(points, normPoints, &H);

	// plane robust estimation
	std::vector<size_t> vec_inliers;
	Sampler sampler;
	if (bFixThreshold) {
		if (maxThreshold == 0)
			maxThreshold = 0.35/H(0,0);
		PlaneSolverAdaptor kernel(normPoints);
		RANSAC(kernel, sampler, vec_inliers, plane, maxThreshold*H(0,0), 0.99, maxIters);
		DEBUG_LEVEL(3, "Robust plane: %u/%u points", vec_inliers.size(), nPoints);
	} else {
		if (maxThreshold != DBL_MAX)
			maxThreshold *= H(0,0);
		PlaneSolverAdaptor kernel(normPoints, 1, 1, 1);
		const std::pair<double,double> ACRansacOut(ACRANSAC(kernel, sampler, vec_inliers, plane, maxThreshold, 0.99, maxIters));
		const double& thresholdSq = ACRansacOut.first;
		maxThreshold = SQRT(thresholdSq)/H(0,0);
		DEBUG_LEVEL(3, "Auto-robust plane: %u/%u points (%g threshold)", vec_inliers.size(), nPoints, maxThreshold);
	}
	unsigned inliers_count = (unsigned)vec_inliers.size();
	if (inliers_count < PlaneSolverAdaptor::MINIMUM_SAMPLES)
		return 0;

	// fit plane to all the inliers
	FitPlaneOnline<TYPE> fitPlane;
	for (unsigned i=0; i<inliers_count; ++i)
		fitPlane.Update(normPoints[vec_inliers[i]]);
	fitPlane.GetPlane(plane);

	// un-normalize plane
	plane.m_fD = (plane.m_fD+plane.m_vN.dot(typename PlaneSolverAdaptor::Model::POINT(H(0,3),H(1,3),H(2,3))))/H(0,0);

	// if a list of inliers is requested, copy it
	if (arrInliers) {
		inliers_count = 0;
		for (unsigned i=0; i<nPoints; ++i)
			if ((arrInliers[i] = (plane.DistanceAbs(points[i]) <= maxThreshold)) == true)
				++inliers_count;
	}
	return inliers_count;
} // TEstimatePlane

} // namespace MVS

// Robustly estimate the plane that fits best the given points
unsigned MVS::EstimatePlane(const Point3dArr& points, Planed& plane, double& maxThreshold, bool arrInliers[], size_t maxIters)
{
	return TEstimatePlane<double,UniformSampler,false>(points, plane, maxThreshold, arrInliers, maxIters);
} // EstimatePlane
// Robustly estimate the plane that fits best the given points, making sure the first point is part of the solution (if any)
unsigned MVS::EstimatePlaneLockFirstPoint(const Point3dArr& points, Planed& plane, double& maxThreshold, bool arrInliers[], size_t maxIters)
{
	return TEstimatePlane<double,UniformSamplerLockFirst,false>(points, plane, maxThreshold, arrInliers, maxIters);
} // EstimatePlaneLockFirstPoint
// Robustly estimate the plane that fits best the given points using a known threshold
unsigned MVS::EstimatePlaneTh(const Point3dArr& points, Planed& plane, double maxThreshold, bool arrInliers[], size_t maxIters)
{
	return TEstimatePlane<double,UniformSampler,true>(points, plane, maxThreshold, arrInliers, maxIters);
} // EstimatePlaneTh
// Robustly estimate the plane that fits best the given points using a known threshold, making sure the first point is part of the solution (if any)
unsigned MVS::EstimatePlaneThLockFirstPoint(const Point3dArr& points, Planed& plane, double maxThreshold, bool arrInliers[], size_t maxIters)
{
	return TEstimatePlane<double,UniformSamplerLockFirst,true>(points, plane, maxThreshold, arrInliers, maxIters);
} // EstimatePlaneThLockFirstPoint
// least squares refinement of the given plane to the 3D point set
int MVS::OptimizePlane(Planed& plane, const Eigen::Vector3d* points, size_t size, int maxIters, double threshold)
{
	return OptimizePlane<double>(plane, points, size, maxIters, threshold);
} // OptimizePlane
/*----------------------------------------------------------------*/

// Robustly estimate the plane that fits best the given points
unsigned MVS::EstimatePlane(const Point3fArr& points, Planef& plane, double& maxThreshold, bool arrInliers[], size_t maxIters)
{
	return TEstimatePlane<float,UniformSampler,false>(points, plane, maxThreshold, arrInliers, maxIters);
} // EstimatePlane
// Robustly estimate the plane that fits best the given points, making sure the first point is part of the solution (if any)
unsigned MVS::EstimatePlaneLockFirstPoint(const Point3fArr& points, Planef& plane, double& maxThreshold, bool arrInliers[], size_t maxIters)
{
	return TEstimatePlane<float,UniformSamplerLockFirst,false>(points, plane, maxThreshold, arrInliers, maxIters);
} // EstimatePlaneLockFirstPoint
// Robustly estimate the plane that fits best the given points using a known threshold
unsigned MVS::EstimatePlaneTh(const Point3fArr& points, Planef& plane, double maxThreshold, bool arrInliers[], size_t maxIters)
{
	return TEstimatePlane<float,UniformSampler,true>(points, plane, maxThreshold, arrInliers, maxIters);
} // EstimatePlaneTh
// Robustly estimate the plane that fits best the given points using a known threshold, making sure the first point is part of the solution (if any)
unsigned MVS::EstimatePlaneThLockFirstPoint(const Point3fArr& points, Planef& plane, double maxThreshold, bool arrInliers[], size_t maxIters)
{
	return TEstimatePlane<float,UniformSamplerLockFirst,true>(points, plane, maxThreshold, arrInliers, maxIters);
} // EstimatePlaneThLockFirstPoint
// least squares refinement of the given plane to the 3D point set
int MVS::OptimizePlane(Planef& plane, const Eigen::Vector3f* points, size_t size, int maxIters, float threshold)
{
	return OptimizePlane<float>(plane, points, size, maxIters, threshold);
} // OptimizePlane
/*----------------------------------------------------------------*/


// estimate the colors of the given dense point cloud
void MVS::EstimatePointColors(const ImageArr& images, PointCloud& pointcloud)
{
	TD_TIMER_START();

	pointcloud.colors.Resize(pointcloud.points.GetSize());
	FOREACH(i, pointcloud.colors) {
		PointCloud::Color& color = pointcloud.colors[i];
		const PointCloud::Point& point = pointcloud.points[i];
		const PointCloud::ViewArr& views= pointcloud.pointViews[i];
		// compute vertex color
		REAL bestDistance(FLT_MAX);
		const Image* pImageData(NULL);
		FOREACHPTR(pView, views) {
			const Image& imageData = images[*pView];
			ASSERT(imageData.IsValid());
			if (imageData.image.empty())
				continue;
			// compute the distance from the 3D point to the image
			const REAL distance(imageData.camera.PointDepth(point));
			ASSERT(distance > 0);
			if (bestDistance > distance) {
				bestDistance = distance;
				pImageData = &imageData;
			}
		}
		if (pImageData == NULL) {
			// set a dummy color
			color = Pixel8U::WHITE;
		} else {
			// get image color
			const Point2f proj(pImageData->camera.ProjectPointP(point));
			color = (pImageData->image.isInsideWithBorder<float,1>(proj) ? pImageData->image.sample(proj) : Pixel8U::WHITE);
		}
	}

	DEBUG_ULTIMATE("Estimate dense point cloud colors: %u colors (%s)", pointcloud.colors.GetSize(), TD_TIMER_GET_FMT().c_str());
} // EstimatePointColors
/*----------------------------------------------------------------*/

// estimates the normals through PCA over the K nearest neighbors
void MVS::EstimatePointNormals(const ImageArr& images, PointCloud& pointcloud, int numNeighbors /*K-nearest neighbors*/)
{
	TD_TIMER_START();

	typedef CGAL::Simple_cartesian<double> kernel_t;
	typedef kernel_t::Point_3 point_t;
	typedef kernel_t::Vector_3 vector_t;
	typedef std::pair<point_t,vector_t> PointVectorPair;
	// fetch the point set
	std::vector<PointVectorPair> pointvectors(pointcloud.points.GetSize());
	FOREACH(i, pointcloud.points)
		reinterpret_cast<Point3d&>(pointvectors[i].first) = pointcloud.points[i];
	// estimates normals direction;
	// Note: pca_estimate_normals() requires an iterator over points
	// as well as property maps to access each point's position and normal.
	#if CGAL_VERSION_NR < 1041301000
	#if CGAL_VERSION_NR < 1040800000
	CGAL::pca_estimate_normals(
	#else
	CGAL::pca_estimate_normals<CGAL::Sequential_tag>(
	#endif
		pointvectors.begin(), pointvectors.end(),
		CGAL::First_of_pair_property_map<PointVectorPair>(),
		CGAL::Second_of_pair_property_map<PointVectorPair>(),
		numNeighbors
	);
	#else
	CGAL::pca_estimate_normals<CGAL::Sequential_tag>(
		pointvectors, numNeighbors,
		CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>())
		.normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>())
	);
	#endif
	// store the point normals
	pointcloud.normals.Resize(pointcloud.points.GetSize());
	FOREACH(i, pointcloud.normals) {
		PointCloud::Normal& normal = pointcloud.normals[i];
		const PointCloud::Point& point = pointcloud.points[i];
		const PointCloud::ViewArr& views= pointcloud.pointViews[i];
		normal = reinterpret_cast<const Point3d&>(pointvectors[i].second);
		// correct normal orientation
		ASSERT(!views.IsEmpty());
		const Image& imageData = images[views.First()];
		if (normal.dot(Cast<float>(imageData.camera.C)-point) < 0)
			normal = -normal;
	}

	DEBUG_ULTIMATE("Estimate dense point cloud normals: %u normals (%s)", pointcloud.normals.GetSize(), TD_TIMER_GET_FMT().c_str());
} // EstimatePointNormals
/*----------------------------------------------------------------*/

bool MVS::EstimateNormalMap(const Matrix3x3f& K, const DepthMap& depthMap, NormalMap& normalMap)
{
	normalMap.create(depthMap.size());
	struct Tool {
		static bool IsDepthValid(Depth d, Depth nd) {
			return nd > 0 && IsDepthSimilar(d, nd, Depth(0.03f));
		}
		// computes depth gradient (first derivative) at current pixel
		static bool DepthGradient(const DepthMap& depthMap, const ImageRef& ir, Point3f& ws) {
			float& w  = ws[0];
			float& wx = ws[1];
			float& wy = ws[2];
			w = depthMap(ir);
			if (w <= 0)
				return false;
			// loop over neighborhood and finding least squares plane,
			// the coefficients of which give gradient of depth
			int whxx(0), whxy(0), whyy(0);
			float wgx(0), wgy(0);
			const int Radius(1);
			int n(0);
			for (int y = -Radius; y <= Radius; ++y) {
				for (int x = -Radius; x <= Radius; ++x) {
					if (x == 0 && y == 0)
						continue;
					const ImageRef pt(ir.x+x, ir.y+y);
					if (!depthMap.isInside(pt))
						continue;
					const float wi(depthMap(pt));
					if (!IsDepthValid(w, wi))
						continue;
					whxx += x*x; whxy += x*y; whyy += y*y;
					wgx += (wi - w)*x; wgy += (wi - w)*y;
					++n;
				}
			}
			if (n < 3)
				return false;
			// solve 2x2 system, generated from depth gradient
			const int det(whxx*whyy - whxy*whxy);
			if (det == 0)
				return false;
			const float invDet(1.f/float(det));
			wx = (float( whyy)*wgx - float(whxy)*wgy)*invDet;
			wy = (float(-whxy)*wgx + float(whxx)*wgy)*invDet;
			return true;
		}
		// computes normal to the surface given the depth and its gradient
		static Normal ComputeNormal(const Matrix3x3f& K, int x, int y, Depth d, Depth dx, Depth dy) {
			ASSERT(ISZERO(K(0,1)));
			return normalized(Normal(
				K(0,0)*dx,
				K(1,1)*dy,
				(K(0,2)-float(x))*dx+(K(1,2)-float(y))*dy-d
			));
		}
	};
	for (int r=0; r<normalMap.rows; ++r) {
		for (int c=0; c<normalMap.cols; ++c) {
			#if 0
			const Depth d(depthMap(r,c));
			if (d <= 0) {
				normalMap(r,c) = Normal::ZERO;
				continue;
			}
			Depth dl, du;
			if (depthMap.isInside(ImageRef(c-1,r-1)) && Tool::IsDepthValid(d, dl=depthMap(r,c-1)) &&  Tool::IsDepthValid(d, du=depthMap(r-1,c)))
				normalMap(r,c) = Tool::ComputeNormal(K, c, r, d, du-d, dl-d);
			else
			if (depthMap.isInside(ImageRef(c+1,r-1)) && Tool::IsDepthValid(d, dl=depthMap(r,c+1)) &&  Tool::IsDepthValid(d, du=depthMap(r-1,c)))
				normalMap(r,c) = Tool::ComputeNormal(K, c, r, d, du-d, d-dl);
			else
			if (depthMap.isInside(ImageRef(c+1,r+1)) && Tool::IsDepthValid(d, dl=depthMap(r,c+1)) &&  Tool::IsDepthValid(d, du=depthMap(r+1,c)))
				normalMap(r,c) = Tool::ComputeNormal(K, c, r, d, d-du, d-dl);
			else
			if (depthMap.isInside(ImageRef(c-1,r+1)) && Tool::IsDepthValid(d, dl=depthMap(r,c-1)) &&  Tool::IsDepthValid(d, du=depthMap(r+1,c)))
				normalMap(r,c) = Tool::ComputeNormal(K, c, r, d, d-du, dl-d);
			else
				normalMap(r,c) = Normal(0,0,-1);
			#else
			// calculates depth gradient at x
			Normal& n = normalMap(r,c);
			if (Tool::DepthGradient(depthMap, ImageRef(c,r), n))
				n = Tool::ComputeNormal(K, c, r, n.x, n.y, n.z);
			else
				n = Normal::ZERO;
			#endif
			ASSERT(normalMap(r,c).dot(K.inv()*Point3f(float(c),float(r),1.f)) <= 0);
		}
	}
	return true;
} // EstimateNormalMap
/*----------------------------------------------------------------*/


// save the depth map in our .dmap file format
bool MVS::SaveDepthMap(const String& fileName, const DepthMap& depthMap)
{
	ASSERT(!depthMap.empty());
	return SerializeSave(depthMap, fileName, ARCHIVE_DEFAULT);
} // SaveDepthMap
/*----------------------------------------------------------------*/
// load the depth map from our .dmap file format
bool MVS::LoadDepthMap(const String& fileName, DepthMap& depthMap)
{
	return SerializeLoad(depthMap, fileName, ARCHIVE_DEFAULT);
} // LoadDepthMap
/*----------------------------------------------------------------*/

// save the normal map in our .nmap file format
bool MVS::SaveNormalMap(const String& fileName, const NormalMap& normalMap)
{
	ASSERT(!normalMap.empty());
	return SerializeSave(normalMap, fileName, ARCHIVE_DEFAULT);
} // SaveNormalMap
/*----------------------------------------------------------------*/
// load the normal map from our .nmap file format
bool MVS::LoadNormalMap(const String& fileName, NormalMap& normalMap)
{
	return SerializeLoad(normalMap, fileName, ARCHIVE_DEFAULT);
} // LoadNormalMap
/*----------------------------------------------------------------*/

// save the confidence map in our .cmap file format
bool MVS::SaveConfidenceMap(const String& fileName, const ConfidenceMap& confMap)
{
	ASSERT(!confMap.empty());
	return SerializeSave(confMap, fileName, ARCHIVE_DEFAULT);
} // SaveConfidenceMap
/*----------------------------------------------------------------*/
// load the confidence map from our .cmap file format
bool MVS::LoadConfidenceMap(const String& fileName, ConfidenceMap& confMap)
{
	return SerializeLoad(confMap, fileName, ARCHIVE_DEFAULT);
} // LoadConfidenceMap
/*----------------------------------------------------------------*/



// export depth map as an image (dark - far depth, light - close depth)
Image8U3 MVS::DepthMap2Image(const DepthMap& depthMap, Depth minDepth, Depth maxDepth)
{
	ASSERT(!depthMap.empty());
	// find min and max values
	if (minDepth == FLT_MAX && maxDepth == 0) {
		cList<Depth,Depth,0> depths(0, depthMap.area());
		for (int i=depthMap.area(); --i >= 0; ) {
			const Depth depth = depthMap[i];
			ASSERT(depth == 0 || depth > 0);
			if (depth > 0)
				depths.Insert(depth);
		}
		if (!depths.empty()) {
			const std::pair<Depth,Depth> th(ComputeX84Threshold<Depth,Depth>(depths.data(), depths.size()));
			const std::pair<Depth,Depth> mm(depths.GetMinMax());
			maxDepth = MINF(th.first+th.second, mm.second);
			minDepth = MAXF(th.first-th.second, mm.first);
		}
		DEBUG_ULTIMATE("\tdepth range: [%g, %g]", minDepth, maxDepth);
	}
	const Depth sclDepth(Depth(1)/(maxDepth - minDepth));
	// create color image
	Image8U3 img(depthMap.size());
	for (int i=depthMap.area(); --i >= 0; ) {
		const Depth depth = depthMap[i];
		img[i] = (depth > 0 ? Pixel8U::gray2color(CLAMP((maxDepth-depth)*sclDepth, Depth(0), Depth(1))) : Pixel8U::BLACK);
	}
	return img;
} // DepthMap2Image
bool MVS::ExportDepthMap(const String& fileName, const DepthMap& depthMap, Depth minDepth, Depth maxDepth)
{
	if (depthMap.empty())
		return false;
	return DepthMap2Image(depthMap, minDepth, maxDepth).Save(fileName);
} // ExportDepthMap
/*----------------------------------------------------------------*/

// export normal map as an image
bool MVS::ExportNormalMap(const String& fileName, const NormalMap& normalMap)
{
	if (normalMap.empty())
		return false;
	Image8U3 img(normalMap.size());
	for (int i=normalMap.area(); --i >= 0; ) {
		img[i] = [](const Normal& n) {
			return ISZERO(n) ?
				Image8U3::Type::BLACK :
				Image8U3::Type(
					CLAMP(ROUND2INT((1.f-n.x)*127.5f), 0, 255),
					CLAMP(ROUND2INT((1.f-n.y)*127.5f), 0, 255),
					CLAMP(ROUND2INT(    -n.z *255.0f), 0, 255)
				);
		} (normalMap[i]);
	}
	return img.Save(fileName);
} // ExportNormalMap
/*----------------------------------------------------------------*/

// export confidence map as an image (dark - low confidence, light - high confidence)
bool MVS::ExportConfidenceMap(const String& fileName, const ConfidenceMap& confMap)
{
	// find min and max values
	FloatArr confs(0, confMap.area());
	for (int i=confMap.area(); --i >= 0; ) {
		const float conf = confMap[i];
		ASSERT(conf == 0 || conf > 0);
		if (conf > 0)
			confs.Insert(conf);
	}
	if (confs.IsEmpty())
		return false;
	const std::pair<float,float> th(ComputeX84Threshold<float,float>(confs.Begin(), confs.GetSize()));
	float minConf = th.first-th.second;
	float maxConf = th.first+th.second;
	if (minConf < 0.1f)
		minConf = 0.1f;
	if (maxConf < 0.1f)
		maxConf = 30.f;
	DEBUG_ULTIMATE("\tconfidence range: [%g, %g]", minConf, maxConf);
	const float deltaConf = maxConf - minConf;
	// save image
	Image8U img(confMap.size());
	for (int i=confMap.area(); --i >= 0; ) {
		const float conf = confMap[i];
		img[i] = (conf > 0 ? (uint8_t)CLAMP((conf-minConf)*255.f/deltaConf, 0.f, 255.f) : 0);
	}
	return img.Save(fileName);
} // ExportConfidenceMap
/*----------------------------------------------------------------*/

// export point cloud
bool MVS::ExportPointCloud(const String& fileName, const Image& imageData, const DepthMap& depthMap, const NormalMap& normalMap)
{
	ASSERT(!depthMap.empty());
	const Camera& P0 = imageData.camera;
	if (normalMap.empty()) {
		// vertex definition
		struct Vertex {
			float x,y,z;
			uint8_t r,g,b;
		};
		// list of property information for a vertex
		static PLY::PlyProperty vert_props[] = {
			{"x", PLY::Float32, PLY::Float32, offsetof(Vertex,x), 0, 0, 0, 0},
			{"y", PLY::Float32, PLY::Float32, offsetof(Vertex,y), 0, 0, 0, 0},
			{"z", PLY::Float32, PLY::Float32, offsetof(Vertex,z), 0, 0, 0, 0},
			{"red", PLY::Uint8, PLY::Uint8, offsetof(Vertex,r), 0, 0, 0, 0},
			{"green", PLY::Uint8, PLY::Uint8, offsetof(Vertex,g), 0, 0, 0, 0},
			{"blue", PLY::Uint8, PLY::Uint8, offsetof(Vertex,b), 0, 0, 0, 0},
		};
		// list of the kinds of elements in the PLY
		static const char* elem_names[] = {
			"vertex"
		};

		// create PLY object
		ASSERT(!fileName.IsEmpty());
		Util::ensureFolder(fileName);
		const size_t bufferSize = depthMap.area()*(8*3/*pos*/+3*3/*color*/+7/*space*/+2/*eol*/) + 2048/*extra size*/;
		PLY ply;
		if (!ply.write(fileName, 1, elem_names, PLY::BINARY_LE, bufferSize))
			return false;

		// describe what properties go into the vertex elements
		ply.describe_property("vertex", 6, vert_props);

		// export the array of 3D points
		Vertex vertex;
		const Point2f scaleImage(static_cast<float>(depthMap.cols)/imageData.image.cols, static_cast<float>(depthMap.rows)/imageData.image.rows);
		for (int j=0; j<depthMap.rows; ++j) {
			for (int i=0; i<depthMap.cols; ++i) {
				const Depth& depth = depthMap(j,i);
				ASSERT(depth >= 0);
				if (depth <= 0)
					continue;
				const Point3f X(P0.TransformPointI2W(Point3(i,j,depth)));
				vertex.x = X.x; vertex.y = X.y; vertex.z = X.z;
				const Pixel8U c(imageData.image.empty() ? Pixel8U::WHITE : imageData.image(ROUND2INT(scaleImage.y*j),ROUND2INT(scaleImage.x*i)));
				vertex.r = c.r; vertex.g = c.g; vertex.b = c.b;
				ply.put_element(&vertex);
			}
		}
		if (ply.get_current_element_count() == 0)
			return false;

		// write to file
		if (!ply.header_complete())
			return false;
	} else {
		// vertex definition
		struct Vertex {
			float x,y,z;
			float nx,ny,nz;
			uint8_t r,g,b;
		};
		// list of property information for a vertex
		static PLY::PlyProperty vert_props[] = {
			{"x", PLY::Float32, PLY::Float32, offsetof(Vertex,x), 0, 0, 0, 0},
			{"y", PLY::Float32, PLY::Float32, offsetof(Vertex,y), 0, 0, 0, 0},
			{"z", PLY::Float32, PLY::Float32, offsetof(Vertex,z), 0, 0, 0, 0},
			{"nx", PLY::Float32, PLY::Float32, offsetof(Vertex,nx), 0, 0, 0, 0},
			{"ny", PLY::Float32, PLY::Float32, offsetof(Vertex,ny), 0, 0, 0, 0},
			{"nz", PLY::Float32, PLY::Float32, offsetof(Vertex,nz), 0, 0, 0, 0},
			{"red", PLY::Uint8, PLY::Uint8, offsetof(Vertex,r), 0, 0, 0, 0},
			{"green", PLY::Uint8, PLY::Uint8, offsetof(Vertex,g), 0, 0, 0, 0},
			{"blue", PLY::Uint8, PLY::Uint8, offsetof(Vertex,b), 0, 0, 0, 0},
		};
		// list of the kinds of elements in the PLY
		static const char* elem_names[] = {
			"vertex"
		};

		// create PLY object
		ASSERT(!fileName.IsEmpty());
		Util::ensureFolder(fileName);
		const size_t bufferSize = depthMap.area()*(8*3/*pos*/+8*3/*normal*/+3*3/*color*/+8/*space*/+2/*eol*/) + 2048/*extra size*/;
		PLY ply;
		if (!ply.write(fileName, 1, elem_names, PLY::BINARY_LE, bufferSize))
			return false;

		// describe what properties go into the vertex elements
		ply.describe_property("vertex", 9, vert_props);

		// export the array of 3D points
		Vertex vertex;
		for (int j=0; j<depthMap.rows; ++j) {
			for (int i=0; i<depthMap.cols; ++i) {
				const Depth& depth = depthMap(j,i);
				ASSERT(depth >= 0);
				if (depth <= 0)
					continue;
				const Point3f X(P0.TransformPointI2W(Point3(i,j,depth)));
				vertex.x = X.x; vertex.y = X.y; vertex.z = X.z;
				const Point3f N(P0.R.t() * Cast<REAL>(normalMap(j,i)));
				vertex.nx = N.x; vertex.ny = N.y; vertex.nz = N.z;
				const Pixel8U c(imageData.image.empty() ? Pixel8U::WHITE : imageData.image(j, i));
				vertex.r = c.r; vertex.g = c.g; vertex.b = c.b;
				ply.put_element(&vertex);
			}
		}
		if (ply.get_current_element_count() == 0)
			return false;

		// write to file
		if (!ply.header_complete())
			return false;
	}
	return true;
} // ExportPointCloud
/*----------------------------------------------------------------*/

//  - IDs are the reference view ID and neighbor view IDs used to estimate the depth-map (global ID)
bool MVS::ExportDepthDataRaw(const String& fileName, const String& imageFileName,
	const IIndexArr& IDs, const cv::Size& imageSize,
	const KMatrix& K, const RMatrix& R, const CMatrix& C,
	Depth dMin, Depth dMax,
	const DepthMap& depthMap, const NormalMap& normalMap, const ConfidenceMap& confMap, const ViewsMap& viewsMap)
{
	ASSERT(IDs.size() > 1 && IDs.size() < 256);
	ASSERT(!depthMap.empty());
	ASSERT(confMap.empty() || depthMap.size() == confMap.size());
	ASSERT(viewsMap.empty() || depthMap.size() == viewsMap.size());
	ASSERT(depthMap.width() <= imageSize.width && depthMap.height() <= imageSize.height);

	FILE* f = fopen(fileName, "wb");
	if (f == NULL) {
		DEBUG("error: opening file '%s' for writing depth-data", fileName.c_str());
		return false;
	}

	// write header
	HeaderDepthDataRaw header;
	header.name = HeaderDepthDataRaw::HeaderDepthDataRawName();
	header.type = HeaderDepthDataRaw::HAS_DEPTH;
	header.imageWidth = (uint32_t)imageSize.width;
	header.imageHeight = (uint32_t)imageSize.height;
	header.depthWidth = (uint32_t)depthMap.cols;
	header.depthHeight = (uint32_t)depthMap.rows;
	header.dMin = dMin;
	header.dMax = dMax;
	if (!normalMap.empty())
		header.type |= HeaderDepthDataRaw::HAS_NORMAL;
	if (!confMap.empty())
		header.type |= HeaderDepthDataRaw::HAS_CONF;
	if (!viewsMap.empty())
		header.type |= HeaderDepthDataRaw::HAS_VIEWS;
	fwrite(&header, sizeof(HeaderDepthDataRaw), 1, f);

	// write image file name
	STATIC_ASSERT(sizeof(String::value_type) == sizeof(char));
	const String FileName(MAKE_PATH_REL(Util::getFullPath(Util::getFilePath(fileName)), Util::getFullPath(imageFileName)));
	const uint16_t nFileNameSize((uint16_t)FileName.length());
	fwrite(&nFileNameSize, sizeof(uint16_t), 1, f);
	fwrite(FileName.c_str(), sizeof(char), nFileNameSize, f);

	// write neighbor IDs
	STATIC_ASSERT(sizeof(uint32_t) == sizeof(IIndex));
	const uint32_t nIDs(IDs.size());
	fwrite(&nIDs, sizeof(IIndex), 1, f);
	fwrite(IDs.data(), sizeof(IIndex), nIDs, f);

	// write pose
	STATIC_ASSERT(sizeof(double) == sizeof(REAL));
	fwrite(K.val, sizeof(REAL), 9, f);
	fwrite(R.val, sizeof(REAL), 9, f);
	fwrite(C.ptr(), sizeof(REAL), 3, f);

	// write depth-map
	fwrite(depthMap.getData(), sizeof(float), depthMap.area(), f);

	// write normal-map
	if ((header.type & HeaderDepthDataRaw::HAS_NORMAL) != 0)
		fwrite(normalMap.getData(), sizeof(float)*3, normalMap.area(), f);

	// write confidence-map
	if ((header.type & HeaderDepthDataRaw::HAS_CONF) != 0)
		fwrite(confMap.getData(), sizeof(float), confMap.area(), f);

	// write views-map
	if ((header.type & HeaderDepthDataRaw::HAS_VIEWS) != 0)
		fwrite(viewsMap.getData(), sizeof(uint8_t)*4, viewsMap.area(), f);

	const bool bRet(ferror(f) == 0);
	fclose(f);
	return bRet;
} // ExportDepthDataRaw

bool MVS::ImportDepthDataRaw(const String& fileName, String& imageFileName,
	IIndexArr& IDs, cv::Size& imageSize,
	KMatrix& K, RMatrix& R, CMatrix& C,
	Depth& dMin, Depth& dMax,
	DepthMap& depthMap, NormalMap& normalMap, ConfidenceMap& confMap, ViewsMap& viewsMap, unsigned flags)
{
	FILE* f = fopen(fileName, "rb");
	if (f == NULL) {
		DEBUG("error: opening file '%s' for reading depth-data", fileName.c_str());
		return false;
	}

	// read header
	HeaderDepthDataRaw header;
	if (fread(&header, sizeof(HeaderDepthDataRaw), 1, f) != 1 ||
		header.name != HeaderDepthDataRaw::HeaderDepthDataRawName() ||
		(header.type & HeaderDepthDataRaw::HAS_DEPTH) == 0 ||
		header.depthWidth <= 0 || header.depthHeight <= 0 ||
		header.imageWidth < header.depthWidth || header.imageHeight < header.depthHeight)
	{
		DEBUG("error: invalid depth-data file '%s'", fileName.c_str());
		return false;
	}

	// read image file name
	STATIC_ASSERT(sizeof(String::value_type) == sizeof(char));
	uint16_t nFileNameSize;
	fread(&nFileNameSize, sizeof(uint16_t), 1, f);
	imageFileName.resize(nFileNameSize);
	fread(imageFileName.data(), sizeof(char), nFileNameSize, f);

	// read neighbor IDs
	STATIC_ASSERT(sizeof(uint32_t) == sizeof(IIndex));
	uint32_t nIDs;
	fread(&nIDs, sizeof(IIndex), 1, f);
	ASSERT(nIDs > 0 && nIDs < 256);
	IDs.resize(nIDs);
	fread(IDs.data(), sizeof(IIndex), nIDs, f);

	// read pose
	STATIC_ASSERT(sizeof(double) == sizeof(REAL));
	fread(K.val, sizeof(REAL), 9, f);
	fread(R.val, sizeof(REAL), 9, f);
	fread(C.ptr(), sizeof(REAL), 3, f);

	// read depth-map
	dMin = header.dMin;
	dMax = header.dMax;
	imageSize.width = header.imageWidth;
	imageSize.height = header.imageHeight;
	if ((flags & HeaderDepthDataRaw::HAS_DEPTH) != 0) {
		depthMap.create(header.depthHeight, header.depthWidth);
		fread(depthMap.getData(), sizeof(float), depthMap.area(), f);
	} else {
		fseek(f, sizeof(float)*header.depthWidth*header.depthHeight, SEEK_CUR);
	}

	// read normal-map
	if ((header.type & HeaderDepthDataRaw::HAS_NORMAL) != 0) {
		if ((flags & HeaderDepthDataRaw::HAS_NORMAL) != 0) {
			normalMap.create(header.depthHeight, header.depthWidth);
			fread(normalMap.getData(), sizeof(float)*3, normalMap.area(), f);
		} else {
			fseek(f, sizeof(float)*3*header.depthWidth*header.depthHeight, SEEK_CUR);
		}
	}

	// read confidence-map
	if ((header.type & HeaderDepthDataRaw::HAS_CONF) != 0) {
		if ((flags & HeaderDepthDataRaw::HAS_CONF) != 0) {
			confMap.create(header.depthHeight, header.depthWidth);
			fread(confMap.getData(), sizeof(float), confMap.area(), f);
		} else {
			fseek(f, sizeof(float)*header.depthWidth*header.depthHeight, SEEK_CUR);
		}
	}

	// read visibility-map
	if ((header.type & HeaderDepthDataRaw::HAS_VIEWS) != 0) {
		if ((flags & HeaderDepthDataRaw::HAS_VIEWS) != 0) {
			viewsMap.create(header.depthHeight, header.depthWidth);
			fread(viewsMap.getData(), sizeof(uint8_t)*4, viewsMap.area(), f);
		}
	}

	const bool bRet(ferror(f) == 0);
	fclose(f);
	return bRet;
} // ImportDepthDataRaw
/*----------------------------------------------------------------*/


// compare the estimated and ground-truth depth-maps
void MVS::CompareDepthMaps(const DepthMap& depthMap, const DepthMap& depthMapGT, uint32_t idxImage, float threshold)
{
	TD_TIMER_START();
	const uint32_t width = (uint32_t)depthMap.width();
	const uint32_t height = (uint32_t)depthMap.height();
	// compute depth errors for each pixel
	cv::resize(depthMapGT, depthMapGT, depthMap.size());
	unsigned nErrorPixels(0);
	unsigned nExtraPixels(0);
	unsigned nMissingPixels(0);
	FloatArr depths(0, depthMap.area());
	FloatArr depthsGT(0, depthMap.area());
	FloatArr errors(0, depthMap.area());
	for (uint32_t i=0; i<height; ++i) {
		for (uint32_t j=0; j<width; ++j) {
			const Depth& depth = depthMap(i,j);
			const Depth& depthGT = depthMapGT(i,j);
			if (depth != 0 && depthGT == 0) {
				++nExtraPixels;
				continue;
			}
			if (depth == 0 && depthGT != 0) {
				++nMissingPixels;
				continue;
			}
			depths.Insert(depth);
			depthsGT.Insert(depthGT);
			const float error(depthGT==0 ? 0 : DepthSimilarity(depthGT, depth));
			errors.Insert(error);
		}
	}
	const float fPSNR((float)ComputePSNR(DMatrix32F((int)depths.size(),1,depths.data()), DMatrix32F((int)depthsGT.size(),1,depthsGT.data())));
	const MeanStd<float,double> ms(errors.data(), errors.size());
	const float mean((float)ms.GetMean());
	const float stddev((float)ms.GetStdDev());
	const std::pair<float,float> th(ComputeX84Threshold<float,float>(errors.data(), errors.size()));
	#if TD_VERBOSE != TD_VERBOSE_OFF
	IDX idxPixel = 0;
	Image8U3 errorsVisual(depthMap.size());
	for (uint32_t i=0; i<height; ++i) {
		for (uint32_t j=0; j<width; ++j) {
			Pixel8U& pix = errorsVisual(i,j);
			const Depth& depth = depthMap(i,j);
			const Depth& depthGT = depthMapGT(i,j);
			if (depth != 0 && depthGT == 0) {
				pix = Pixel8U::GREEN;
				continue;
			}
			if (depth == 0 && depthGT != 0) {
				pix = Pixel8U::BLUE;
				continue;
			}
			const float error = errors[idxPixel++];
			if (depth == 0 && depthGT == 0) {
				pix = Pixel8U::BLACK;
				continue;
			}
			if (error > threshold) {
				pix = Pixel8U::RED;
				++nErrorPixels;
				continue;
			}
			const uint8_t gray((uint8_t)CLAMP((1.f-SAFEDIVIDE(ABS(error), threshold))*255.f, 0.f, 255.f));
			pix = Pixel8U(gray, gray, gray);
		}
	}
	errorsVisual.Save(ComposeDepthFilePath(idxImage, "errors.png"));
	#endif
	VERBOSE("Depth-maps compared for image % 3u: %.4f PSNR; %g median %g mean %g stddev error; %u (%.2f%%%%) error %u (%.2f%%%%) missing %u (%.2f%%%%) extra pixels (%s)",
		idxImage,
		fPSNR,
		th.first, mean, stddev,
		nErrorPixels, (float)nErrorPixels*100.f/depthMap.area(),
		nMissingPixels, (float)nMissingPixels*100.f/depthMap.area(),
		nExtraPixels, (float)nExtraPixels*100.f/depthMap.area(),
		TD_TIMER_GET_FMT().c_str()
	);
}

// compare the estimated and ground-truth normal-maps
void MVS::CompareNormalMaps(const NormalMap& normalMap, const NormalMap& normalMapGT, uint32_t idxImage)
{
	TD_TIMER_START();
	// load normal data
	const uint32_t width = (uint32_t)normalMap.width();
	const uint32_t height = (uint32_t)normalMap.height();
	// compute normal errors for each pixel
	cv::resize(normalMapGT, normalMapGT, normalMap.size());
	FloatArr errors(0, normalMap.area());
	for (uint32_t i=0; i<height; ++i) {
		for (uint32_t j=0; j<width; ++j) {
			const Normal& normal = normalMap(i,j);
			const Normal& normalGT = normalMapGT(i,j);
			if (normal != Normal::ZERO && normalGT == Normal::ZERO)
				continue;
			if (normal == Normal::ZERO && normalGT != Normal::ZERO)
				continue;
			if (normal == Normal::ZERO && normalGT == Normal::ZERO) {
				errors.Insert(0.f);
				continue;
			}
			ASSERT(ISEQUAL(norm(normal),1.f) && ISEQUAL(norm(normalGT),1.f));
			const float error(FR2D(ACOS(CLAMP(normal.dot(normalGT), -1.f, 1.f))));
			errors.Insert(error);
		}
	}
	const MeanStd<float,double> ms(errors.Begin(), errors.GetSize());
	const float mean((float)ms.GetMean());
	const float stddev((float)ms.GetStdDev());
	const std::pair<float,float> th(ComputeX84Threshold<float,float>(errors.Begin(), errors.GetSize()));
	VERBOSE("Normal-maps compared for image % 3u: %.2f median %.2f mean %.2f stddev error (%s)",
		idxImage,
		th.first, mean, stddev,
		TD_TIMER_GET_FMT().c_str()
	);
}
/*----------------------------------------------------------------*/
