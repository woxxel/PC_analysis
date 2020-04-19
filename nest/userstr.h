//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//            Nested Sampling
// 
// Filename:  userstr.h
// 
// Purpose:   Define "UserObjectStr" substructure of object, and error codes.
//
// History:   JS     4 Sep 2004
//-----------------------------------------------------------------------------
#ifndef USERSTRH
#define USERSTRH

typedef struct         // <<<< PARAMETERS COMMON TO ALL OBJECTS >>>>
{
    int       Ndim;    // I    input dimension of prior sphere
    double    Radius;  // I    input radius of prior sphere
} ParamStr;

typedef struct         // <<<< USER OBJECT INFORMATION >>>>
{
    double*   x;       //   O  individual positional coordinates   [Ndim]
    ParamStr* Param;   // I    parameters common to all objects
} UserObjectStr;

#undef  E_ANALYSIS          // Range of analytic check-results exceeded
#define E_ANALYSIS  -33

#endif

