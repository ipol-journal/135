/** 
 * @file nninterp.h
 * @brief Nearest neighbor image interpolation
 * @author Pascal Getreuer <getreuer@gmail.com>
 */
#ifndef _NNINTERP_H_
#define _NNINTERP_H_
#include "imageio.h"

void NearestInterp(uint32_t *Output, int OutputWidth, int OutputHeight,
    uint32_t *Input, int InputWidth, int InputHeight, 
    float ScaleFactor, int CenteredGrid);

#endif /* _NNINTERP_H_ */
