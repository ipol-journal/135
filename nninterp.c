/** 
 * @file nninterp.c
 * @brief Nearest neighbor image interpolation
 * @author Pascal Getreuer <getreuer@gmail.com>
 */

#include <math.h>
#include "nninterp.h"


void NearestInterp(uint32_t *Output, int OutputWidth, int OutputHeight,
    uint32_t *Input, int InputWidth, int InputHeight, 
    float ScaleFactor, int CenteredGrid)
{
    const float Start  = (CenteredGrid) ? (1/ScaleFactor - 1)/2 : 0;
    int x, y, ix, iy;

    
    for(y = 0; y < OutputHeight; y++, Output += OutputWidth)
    {
        iy = (int)floor(Start + y/ScaleFactor + 0.5);
        
        if(iy < 0)
            iy = 0;
        else if(iy >= InputHeight)
            iy = InputHeight - 1;
        
        for(x = 0; x < OutputWidth; x++)
        {
            ix = (int)floor(Start + x/ScaleFactor + 0.5);
            
            if(ix < 0)
                ix = 0;
            else if(ix >= InputWidth)
                ix = InputWidth - 1;
                        
            Output[x] = Input[ix + InputWidth*iy];                      
        }
    }
}
