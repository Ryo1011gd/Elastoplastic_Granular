//================================================================================================//
//------------------------------------------------------------------------------------------------//
//              GranularFlowSimulation-SPH (Full-Explicit)                                        //
//------------------------------------------------------------------------------------------------//
//    Copyright       : Ryo Yokoyama                                                              //
//    OpenACC GPU     : 2025                                                                      //
//    HPCSDK 25.9 CUDA version 13.0                                                            	  //
//    MPI-OpenACC hybrid Parallel Computation					                  //
//    For HPCSDK https://developer.nvidia.com/nvidia-hpc-sdk-241-download                         //
//    For MPI   https://www.open-mpi.org/software/ompi/v4.1/					  //
//    Final Check     : 2025 22 October                                                           //
//    												                                              //
//                                                                                                //
//================================================================================================//
//        PLEASE DO NOT DISTRIBUTE THIS CODE TO OUTSIDE OF OKAMOTO LABO                           //
//                                                                                                //
//================================================================================================//
//                      HOW TO COMPILE IN LINUX SYSTEM                                            //
//                   1. /Elastoplastic_Granular/generator/make/                                   //
//                   2. /Elastoplastic_Granular/source/make                                       //
//                   3. /Elastoplastic_Granular/results/and ./generate.sh  and ./execute.sh       //
//================================================================================================//
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <assert.h>
#include <iostream>
#include <vector>
#include <omp.h>
#include "errorfunc.h"
#include "log.h"
#include <mutex>
#include <openacc.h>
#ifdef _CUDA
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cuda_runtime_api.h>
#endif

const double DOUBLE_ZERO[32]={0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0};

using namespace std;
//#define TWO_DIMENSIONAL //If you want to perform 3D simulation, please comment out //
#define _OPENMP
#define DIM 3
#define CUDA
//#define FLUID         // this is for fluid calculation coupling. please comment out if you want to perform only elastoplastic simulation //
#define ELASTOPLASTIC
#define MELTING
#define CORE_NUMBER   16 //please change the core numbers of your CPU for OpenMP


// Property definition
#define TYPE_COUNT   6
#define FLUID_BEGIN  0
#define FLUID_END    2
#define STRUCTURE_BEGIN 2
#define STRUCTURE_END   4
#define WALL_BEGIN   4
#define WALL_END     6

#define  DEFAULT_LOG  "sample.log"
#define  DEFAULT_DATA "sample.data"
#define  DEFAULT_GRID "sample.grid"
#define  DEFAULT_PROF "sample%03d.prof"
#define  DEFAULT_VTK  "sample%03d.vtk"

// Calculation and Output
static double ParticleSpacing=0.0;
static double ParticleVolume=0.0;
static double OutputInterval=0.0;
static double OutputNext=0.0;
static double VtkOutputInterval=0.0;
static double VtkOutputNext=0.0;
static double EndTime=0.0;
static double Time=0.0;
static double Dt=1.0e100;
static double Elastic_Dt=1.0e100;
static double DomainMin[DIM];
static double DomainMax[DIM];
static double DomainWidth[DIM];
#pragma acc declare create(ParticleSpacing,ParticleVolume,Dt,Elastic_Dt,DomainMin,DomainMax,DomainWidth)

#define Mod(x,w) ((x)-(w)*floor((x)/(w)))   // mod 

#define MAX_NEIGHBOR_COUNT 512
// Particle
static int ParticleCount;
static int *Property;                     // particle type
static double (*Mass);                    // mass
static double (*Position)[DIM];
static double (*InitialPosition)[DIM];
static double (*Velocity)[DIM];           // momentum
static double (*Force)[DIM];              // total explicit force acting on the particle
static int *NeighborCount;                   // [ParticleCount]
static int (*Neighbor)[MAX_NEIGHBOR_COUNT];  // [ParticleCount]
static double (*NeighborCalculatedPosition)[DIM];
#define MARGIN (0.1*ParticleSpacing)
#pragma acc declare create(ParticleCount,Property,Mass,Position,InitialPosition,Velocity,Force,NeighborCount,Neighbor,NeighborCalculatedPosition)


// BackGroundCells
#ifdef TWO_DIMENSIONAL
#define CellId(iCX,iCY,iCZ)  (((iCX)%CellCount[0]+CellCount[0])%CellCount[0]*CellCount[1] + ((iCY)%CellCount[1]+CellCount[1])%CellCount[1])
#else
#define CellId(iCX,iCY,iCZ)  (((iCX)%CellCount[0]+CellCount[0])%CellCount[0]*CellCount[1]*CellCount[2] + ((iCY)%CellCount[1]+CellCount[1])%CellCount[1]*CellCount[2] + ((iCZ)%CellCount[2]+CellCount[2])%CellCount[2])
#endif

static int PowerParticleCount;
static int ParticleCountPower;                   
static double CellWidth = 0.0;
static int CellCount[DIM];
static int CellCounts = 0;
static int *CellParticleBegin;  // beginning of particles in the cell
static int *CellParticleEnd;    // number of particles in the cell
static int *CellIndex;  // [ParticleCountPower>>1]
static int *CellParticle;       // array of particle id in the cells) [ParticleCountPower>>1]
#pragma acc declare create(PowerParticleCount,ParticleCountPower,CellWidth,CellCount,CellCounts,CellParticleBegin,CellParticleEnd,CellIndex,CellParticle)

// Type
static double Density[TYPE_COUNT];
static double BulkModulus[TYPE_COUNT];
static double BulkViscosity[TYPE_COUNT];
static double ShearViscosity[TYPE_COUNT];
static double SurfaceTension[TYPE_COUNT];
static double CofA[TYPE_COUNT];   // coefficient for attractive pressure
static double CofK;               // coefficinet (ratio) for diffuse interface thickness normalized by ParticleSpacing
static double InteractionRatio[TYPE_COUNT][TYPE_COUNT];
#pragma acc declare create(Density,BulkModulus,BulkViscosity,ShearViscosity,SurfaceTension,CofA,CofK,InteractionRatio)


// Fluid
static int FluidParticleBegin;
static int FluidParticleEnd;
static double *DensityA;        // number density per unit volume for attractive pressure
static double (*GravityCenter)[DIM];
static double *PressureA;       // attractive pressure (surface tension)
static double *VolStrainP;        // number density per unit volume for base pressure
static double *DivergenceP;     // volumetric strainrate for pressure B
static double *PressureP;       // base pressure
static double *VirialPressureAtParticle; // VirialPressureInSingleParticleRegion
static double (*VirialStressAtParticle)[DIM][DIM];
static double *Mu;              // viscosity coefficient for shear
static double *Lambda;          // viscosity coefficient for bulk
static double *Kappa;           // bulk modulus
#pragma acc declare create(DensityA,GravityCenter,PressureA,VolStrainP,DivergenceP,PressureP,VirialPressureAtParticle,VirialStressAtParticle,Mu,Lambda,Kappa)

static double Gravity[DIM] = {0.0,0.0,0.0};
#pragma acc declare create(Gravity)

// Wall
static int WallParticleBegin;
static int WallParticleEnd;
static double WallCenter[WALL_END][DIM];
static double WallVelocity[WALL_END][DIM];
static double WallOmega[WALL_END][DIM];
static double WallRotation[WALL_END][DIM][DIM];
#pragma acc declare create(WallCenter,WallVelocity,WallOmega,WallRotation)

//Structure
static double YoungModulus[TYPE_COUNT];
static double PoissonRatio[TYPE_COUNT];
static double Cohesion[TYPE_COUNT];
static double InternalFrictionAngle[TYPE_COUNT];
static double DilatancyFrictionAngle[TYPE_COUNT];
static int StructureParticleBegin;
static int StructureParticleEnd;
static double *Young;
static double *LambdaLames;
static double *MuLames;
static double (*Strain)[DIM][DIM];
static double (*Spin)[DIM][DIM];
static double (*PlasticStrainRate)[DIM][DIM];  //added for the elastic
static double (*Stress)[DIM][DIM];
static double (*Acceleration)[DIM];
static double (*DiffusiveCoefficient);
static double (*ShearRate);
#pragma acc declare create(Young,YoungModulus,PoissonRatio,Cohesion,InternalFrictionAngle,DilatancyFrictionAngle,LambdaLames,MuLames)
#pragma acc declare create(Strain,Spin,PlasticStrainRate,Stress,Acceleration,DiffusiveCoefficient,ShearRate)

//Energy COnservation Equation

static double Heat[TYPE_COUNT];
static double ThermalConductivity[TYPE_COUNT];
static double MeltingPoint[TYPE_COUNT];
static double SpecificHeat[TYPE_COUNT];
static double SolidifyingEnthalpy[TYPE_COUNT];
static double LiquefyingEnthalpy[TYPE_COUNT];
static double CriticalSolidFraction[TYPE_COUNT];
#pragma acc declare create(Heat,ThermalConductivity,MeltingPoint,SpecificHeat,SolidifyingEnthalpy,LiquefyingEnthalpy,CriticalSolidFraction)

//Parameters for melting and solidifcation //
static double *Temperature;
static double *Enthalpy;
static double *SolidFraction;   //How amount of solid is contained in one particle
static double *Conductivity;
static double *MeltingTemp;
static double *Cp;
static double *H0;
static double *H1;
#pragma acc declare create(Temperature,Enthalpy,SolidFraction)
#pragma acc declare create(Conductivity,MeltingTemp,Cp,H0,H1)



// proceedures
static void readDataFile(char *filename);
static void readGridFile(char *filename);
static void writeProfFile(char *filename);
static void writeVtkFile(char *filename);
static void initializeWeight( void );
static void initializeDomain( void );
static void initializeFluid( void );
static void initializeWall( void );
static void initializeStructure( void );
static void calculateConvection();
static void calculateWall();
static void calculatePeriodicBoundary();
static void resetForce();
static int neighborCalculation();
static void calculateNeighbor();
static void calculatePhysicalCoefficients();
static void calculateDensityA();
static void calculatePressureA();
static void calculateGravityCenter();
static void calculateDiffuseInterface();
static void calculateDensityP();
static void calculateDivergenceP();
static void calculatePressureP();
static void calculateViscosityV();
static void calculateGravity();
static void calculateAcceleration();
static void calculateVirialPressureAtParticle();
static void calculateVirialStressAtParticle();

//Elastoplastic
static void calculateLamesconstant();
static void resetAcceleration();
static void calculateStressForce();
static void calculateInterfaceForce();
static void calculateInterfaceViscosity();
static void selectFreeGPU();
static void calculateStrainRateTensor();
static void calculateSpinTensor();
static void calculateStress();
static void calculatePlasticStrainRateTensor();
static void calculateDiffusive();
static void updateElasticForce();
static void updateElasticPosition();

//EnergyConservation
static void calculateEnergyConservation();
static void calculateTemperature();
static void calculateSolidFraction();
static void calculateViscosity();


// dual kernel functions
static double RadiusRatioA;
static double RadiusRatioG;
static double RadiusRatioP;
static double RadiusRatioV;

static double MaxRadius = 0.0;
static double RadiusA = 0.0;
static double RadiusG = 0.0;
static double RadiusP = 0.0;
static double RadiusV = 0.0;
static double Swa = 1.0;
static double Swg = 1.0;
static double Swp = 1.0;
static double Swv = 1.0;
static double N0a = 1.0;
static double N0p = 1.0;
static double R2g = 1.0;

#pragma acc declare create(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)


#pragma acc routine seq
static double wa(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swa * 1.0/(h*h) * (r/h)*(1.0-(r/h))*(1.0-(r/h));
#else
    return 1.0/Swa * 1.0/(h*h*h) * (r/h)*(1.0-(r/h))*(1.0-(r/h));
#endif
}

#pragma acc routine seq
static double dwadr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swa * 1.0/(h*h) * (1.0-(r/h))*(1.0-3.0*(r/h))*(1.0/h);
#else
    return 1.0/Swa * 1.0/(h*h*h) * (1.0-(r/h))*(1.0-3.0*(r/h))*(1.0/h);
#endif
}

#pragma acc routine seq
static double wg(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swg * 1.0/(h*h) * ((1.0-r/h)*(1.0-r/h));
#else
    return 1.0/Swg * 1.0/(h*h*h) * ((1.0-r/h)*(1.0-r/h));
#endif
}

#pragma acc routine seq
static double dwgdr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swg * 1.0/(h*h) * (-2.0/h*(1.0-r/h));
#else
    return 1.0/Swg * 1.0/(h*h*h) * (-2.0/h*(1.0-r/h));
#endif
}

#pragma acc routine seq
static double wp(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swp * 1.0/(h*h) * ((1.0-r/h)*(1.0-r/h));
#else    
    return 1.0/Swp * 1.0/(h*h*h) * ((1.0-r/h)*(1.0-r/h));
#endif
}

#pragma acc routine seq
static double dwpdr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swp * 1.0/(h*h) * (-2.0/h*(1.0-r/h));
#else
    return 1.0/Swp * 1.0/(h*h*h) * (-2.0/h*(1.0-r/h));
#endif
}

#pragma acc routine seq
static double wv(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swv * 1.0/(h*h) * ((1.0-r/h)*(1.0-r/h));
#else    
    return 1.0/Swv * 1.0/(h*h*h) * ((1.0-r/h)*(1.0-r/h));
#endif
}

#pragma acc routine seq
static double dwvdr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swv * 1.0/(h*h) * (-2.0/h*(1.0-r/h));
#else
    return 1.0/Swv * 1.0/(h*h*h) * (-2.0/h*(1.0-r/h));
#endif
}


	clock_t cFrom, cTill, cStart, cEnd;
	clock_t cNeigh=0, cExplicit=0, cVirial=0, cOther=0;



const double L = 20.0e-2;        // Length in meters (20 cm)
const double k = 1.875 / L;      // k value for first vibration mode

// Function to compute f(x1)
double compute_fx1(double x1) {
    double kL = k * L;
    double kx1 = k * x1;

    double term1 = (cos(kL) + cosh(kL)) * (sin(kx1) - sinh(kx1));
    double term2 = (sin(kL) + sinh(kL)) * (cos(kx1) - cosh(kx1));

    return term1 - term2;
}




int main(int argc, char *argv[])
{
	
    char logfilename[1024]  = DEFAULT_LOG;
    char datafilename[1024] = DEFAULT_DATA;
    char gridfilename[1024] = DEFAULT_GRID;
    char proffilename[1024] = DEFAULT_PROF;
    char vtkfilename[1024] = DEFAULT_VTK;
	    int numberofthread = 1;
    
    {
        if(argc>1)strcpy(datafilename,argv[1]);
        if(argc>2)strcpy(gridfilename,argv[2]);
        if(argc>3)strcpy(proffilename,argv[3]);
        if(argc>4)strcpy(vtkfilename,argv[4]);
        if(argc>5)strcpy( logfilename,argv[5]);
    	if(argc>6)numberofthread=atoi(argv[6]);
    }
   // selectFreeGPU();
    
    log_open(logfilename);
    {
        time_t t=time(NULL);
        log_printf("start reading files at %s\n",ctime(&t));
    }
	{
		#ifdef _OPENMP
		omp_set_num_threads( CORE_NUMBER );
		#pragma omp parallel
		{
			if(omp_get_thread_num()==0){
				log_printf("omp_get_num_threads()=%d\n", omp_get_num_threads() );
			}
		}
		#endif
    }
    readDataFile(datafilename);
    readGridFile(gridfilename);
    {
        time_t t=time(NULL);
        log_printf("start initialization at %s\n",ctime(&t));
    }
    initializeWeight();
    initializeFluid();
    initializeWall();
    initializeDomain();

//	#pragma acc enter data create(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
//	#pragma acc enter data create(WallCenter[0:WALL_END][0:DIM],WallVelocity[0:WALL_END][0:DIM],WallOmega[0:WALL_END][0:DIM],WallRotation[0:WALL_END][0:DIM][0:DIM])
//	#pragma acc enter data create(PowerParticleCount,ParticleCountPower,CellWidth,CellCount[0:DIM],CellParticleBegin[0:CellCounts],CellParticleEnd[0:CellCounts])
//	#pragma acc enter data create(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount])
//	#pragma acc enter data create(Force[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],NeighborCalculatedPosition[0:ParticleCount][0:DIM])
//	#pragma acc enter data create(DensityA[0:ParticleCount],GravityCenter[0:ParticleCount][0:DIM],PressureA[0:ParticleCount])
//	#pragma acc enter data create(VolStrainP[0:ParticleCount],DivergenceP[0:ParticleCount],PressureP[0:ParticleCount])
//	#pragma acc enter data create(VirialPressureAtParticle[0:ParticleCount],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	
	// data transfer from host to device
	#pragma acc update device(ParticleSpacing,ParticleVolume,Dt,Elastic_Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
	#pragma acc update device(ParticleCount,Property[0:ParticleCount],Mass[0:ParticleCount],Position[0:ParticleCount][0:DIM],InitialPosition[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM])
	#pragma acc update device(Density[0:TYPE_COUNT],BulkModulus[0:TYPE_COUNT],BulkViscosity[0:TYPE_COUNT],ShearViscosity[0:TYPE_COUNT],SurfaceTension[0:TYPE_COUNT])
	#pragma acc update device(CofA[0:TYPE_COUNT],CofK,InteractionRatio[0:TYPE_COUNT][0:TYPE_COUNT])
	#pragma acc update device(Mu[0:ParticleCount],Lambda[0:ParticleCount],Kappa[0:ParticleCount],Gravity[0:DIM])
	#pragma acc update device(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
	#pragma acc update device(WallCenter[0:WALL_END][0:DIM],WallVelocity[0:WALL_END][0:DIM],WallOmega[0:WALL_END][0:DIM],WallRotation[0:WALL_END][0:DIM][0:DIM])
	#pragma acc update device(PowerParticleCount,ParticleCountPower,CellWidth,CellCount[0:DIM],CellParticleBegin[0:CellCounts],CellParticleEnd[0:CellCounts])

	//#ifdef ELASTOPLASTIC
	#pragma acc update device(YoungModulus[0:TYPE_COUNT],PoissonRatio[0:TYPE_COUNT],Cohesion[0:TYPE_COUNT],InternalFrictionAngle[0:TYPE_COUNT],DilatancyFrictionAngle[0:TYPE_COUNT])
	#pragma acc update device(Young[0:ParticleCount],LambdaLames[0:ParticleCount],MuLames[0:ParticleCount])
	#pragma acc update device(Strain[0:ParticleCount][0:DIM][0:DIM],Spin[0:ParticleCount][0:DIM][0:DIM],PlasticStrainRate[0:ParticleCount][0:DIM][0:DIM],Stress[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc update device(Acceleration[0:ParticleCount][0:DIM],DiffusiveCoefficient[0:ParticleCount],ShearRate[0:ParticleCount])
	//#endif

	#ifdef MELTING
    #pragma acc update device(Temperature[0:ParticleCount],Enthalpy[0:ParticleCount],SolidFraction[0:ParticleCount])
    #pragma acc update device(MeltingTemp[0:ParticleCount],Cp[0:ParticleCount],H0[0:ParticleCount],H1[0:ParticleCount])
     #pragma acc update device(ThermalConductivity[0:TYPE_COUNT],SpecificHeat[0:TYPE_COUNT],MeltingPoint[0:TYPE_COUNT],SolidifyingEnthalpy[0:TYPE_COUNT],LiquefyingEnthalpy[0:TYPE_COUNT],CriticalSolidFraction[0:TYPE_COUNT])
    #endif


	{
	calculateNeighbor();
	calculateDensityA();
	calculateGravityCenter();
	calculateDensityP();   
	calculateLamesconstant();   
    calculateStrainRateTensor();
    calculateSpinTensor();
    calculatePlasticStrainRateTensor();
    calculateStress();
	writeVtkFile("output.vtk");
		
	{
		time_t t=time(NULL);
		log_printf("start main roop at %s\n",ctime(&t));
	}
	int iStep=(int)(Time/Dt);
	cStart = clock();
	cFrom = cStart;
	while(Time < EndTime + 1.0e-5*Dt){
			
		if( Time + 1.0e-5*Dt >= OutputNext ){
				char filename[256];
				sprintf(filename,proffilename,iStep);
				writeProfFile(filename);
				log_printf("@ Prof Output Time : %e\n", Time );
				OutputNext += OutputInterval;
		}
			cTill = clock(); cOther += (cTill-cFrom); cFrom = cTill;
			
			// wall calculation
			calculateWall();
			
			// periodic boundary calculation
			calculatePeriodicBoundary();
			
			// reset Force to calculate conservative interaction
			resetForce();
			resetAcceleration();
		
		
			cTill = clock(); cExplicit += (cTill-cFrom); cFrom = cTill;
			
			// calculate Neighbor
			//if(neighborCalculation()==1){
			calculateNeighbor();
			//}
			cTill = clock(); cNeigh += (cTill-cFrom); cFrom = cTill;
			
			#ifdef FLUID
			// calculate density
			calculateDensityA();
			calculateGravityCenter();
			calculateDensityP();
			calculateDivergenceP();
			#endif
			
			// calculate physical coefficient (viscosity, bulk modulus, bulk viscosity..)
			calculatePhysicalCoefficients();
			
		

			#ifdef FLUID
			// calculate pressure 
	        calculatePressureP();
			
			// calculate P(s,rho) s:fixed
		  	calculatePressureA();
			
			// calculate diffuse interface force
			calculateDiffuseInterface();
			
		    // calculate shear viscosity
			calculateViscosityV();
			#endif

			#ifdef MELTING
		   // calculate Enrgy conservation equation
      	  //  calculateEnergyConservation();
     
          	//calculate solid fraction
      	  //  calculateSolidFraction();

   
            //calculate phase change
      	 //   calculateViscosity();
      	   #endif
      	  

			calculateGravity();
			
            // calculate intermidiate Velocity
            calculateAcceleration();    		
				 
           
			 calculateConvection();



   			int substeps = (int)(Dt / Elastic_Dt); 

            #ifdef ELASTOPLASTIC
  			for (int substep = 0; substep < substeps; ++substep) {

			//calculate strain rate
            calculateStrainRateTensor();

			//calculate spin tensor
            calculateSpinTensor();
    
			//calculate plastic deformation
            calculatePlasticStrainRateTensor();
    
			//update the stress
            calculateStress();
    
			//diffusive term for the stabilization 
           // calculateDiffusive();

			//update the force
            calculateStressForce();

			//update the velocity
            updateElasticForce();
			
    
			//update the acceleration
            updateElasticPosition();

  			}
			#endif
           

			cTill = clock(); cExplicit += (cTill-cFrom); cFrom = cTill;
			
			
			if( Time + 1.0e-5*Dt >= VtkOutputNext ){
				calculateVirialStressAtParticle();
				cTill = clock(); cVirial += (cTill-cFrom); cFrom = cTill;

				char filename [256];
				sprintf(filename,vtkfilename,iStep);
				writeVtkFile(filename);
				log_printf("@ Vtk Output Time : %e\n", Time );
				VtkOutputNext += VtkOutputInterval;
				cTill = clock(); cOther += (cTill-cFrom); cFrom = cTill;

			}
			
			Time += Dt;
			iStep++;
			cTill = clock(); cExplicit += (cTill-cFrom); cFrom = cTill;
		}
	}
	cEnd = cTill;
	
	{
		time_t t=time(NULL);
		log_printf("end main roop at %s\n",ctime(&t));
		log_printf("neighbor search:         %lf [CPU sec]\n", (double)cNeigh/CLOCKS_PER_SEC);
		log_printf("explicit calculation:    %lf [CPU sec]\n", (double)cExplicit/CLOCKS_PER_SEC);
		log_printf("virial calculation:      %lf [CPU sec]\n", (double)cVirial/CLOCKS_PER_SEC);
		log_printf("other calculation:       %lf [CPU sec]\n", (double)cOther/CLOCKS_PER_SEC);
		log_printf("total:                   %lf [CPU sec]\n", (double)(cNeigh+cExplicit+cVirial+cOther)/CLOCKS_PER_SEC);
		log_printf("total (check):           %lf [CPU sec]\n", (double)(cEnd-cStart)/CLOCKS_PER_SEC);
	}
	
	
	#pragma acc exit data delete(ParticleCount,ParticleSpacing,ParticleVolume,Dt,Elastic_Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
	#pragma acc exit data delete(Property[0:ParticleCount],Mass[0:ParticleCount],Position[0:ParticleCount][0:DIM],InitialPosition[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM])
	#pragma acc exit data delete(Density[0:TYPE_COUNT],BulkModulus[0:TYPE_COUNT],BulkViscosity[0:TYPE_COUNT],ShearViscosity[0:TYPE_COUNT],SurfaceTension[0:TYPE_COUNT])
	#pragma acc exit data delete(CofA[0:TYPE_COUNT],CofK,InteractionRatio[0:TYPE_COUNT][0:TYPE_COUNT])
	#pragma acc exit data delete(Mu[0:ParticleCount],Lambda[0:ParticleCount],Kappa[0:ParticleCount],Gravity[0:DIM])
	#pragma acc exit data delete(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
//	#pragma acc exit data delete(FluidParticleBegin,FluidParticleEnd,WallParticleBegin,WallParticleEnd)
	#pragma acc exit data delete(WallCenter[0:WALL_END][0:DIM],WallVelocity[0:WALL_END][0:DIM],WallOmega[0:WALL_END][0:DIM],WallRotation[0:WALL_END][0:DIM][0:DIM])
	#pragma acc exit data delete(PowerParticleCount,ParticleCountPower,CellWidth,CellCount[0:DIM])
	#pragma acc exit data delete(CellParticleBegin[0:CellCounts],CellParticleEnd[0:CellCounts])
	#pragma acc exit data delete(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount])
	#pragma acc exit data delete(Force[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],NeighborCalculatedPosition[0:ParticleCount][0:DIM])
	#pragma acc exit data delete(DensityA[0:ParticleCount],GravityCenter[0:ParticleCount][0:DIM],PressureA[0:ParticleCount])
	#pragma acc exit data delete(VolStrainP[0:ParticleCount],DivergenceP[0:ParticleCount],PressureP[0:ParticleCount])
	#pragma acc exit data delete(VirialPressureAtParticle[0:ParticleCount],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc exit data delete(YoungModulus[0:TYPE_COUNT],PoissonRatio[0:TYPE_COUNT],Cohesion[0:TYPE_COUNT],InternalFrictionAngle[0:TYPE_COUNT],DilatancyFrictionAngle[0:TYPE_COUNT])
	#pragma acc exit data delete(Young[0:ParticleCount],LambdaLames[0:ParticleCount],MuLames[0:ParticleCount])
	#pragma acc exit data delete(Strain[0:ParticleCount][0:DIM][0:DIM],PlasticStrainRate[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc exit data delete(Stress[0:ParticleCount][0:DIM][0:DIM],Acceleration[0:ParticleCount][0:DIM],DiffusiveCoefficient[0:ParticleCount],ShearRate[0:ParticleCount])

	
	return 0;
	
}

static void readDataFile(char *filename)
{
    FILE * fp;
    char buf[1024];
    const int reading_global=0;
    int mode=reading_global;
    

 
    fp=fopen(filename,"r");
    mode=reading_global;
    while(fp!=NULL && !feof(fp) && !ferror(fp)){
        if(fgets(buf,sizeof(buf),fp)!=NULL){
            if(buf[0]=='#'){}
       else if(sscanf(buf," Dt %lf",&Dt)==1){mode=reading_global;}
       else if(sscanf(buf," ElasticDt %lf",&Elastic_Dt)==1){mode=reading_global;}
       else if(sscanf(buf," OutputInterval %lf",&OutputInterval)==1){mode=reading_global;}
       else if(sscanf(buf," VtkOutputInterval %lf",&VtkOutputInterval)==1){mode=reading_global;}
       else if(sscanf(buf," EndTime %lf",&EndTime)==1){mode=reading_global;}
       else if(sscanf(buf," RadiusRatioA %lf",&RadiusRatioA)==1){mode=reading_global;}
        	// else if(sscanf(buf," RadiusRatioG %lf",&RadiusRatioG)==1){mode=reading_global;}
       else if(sscanf(buf," RadiusRatioP %lf",&RadiusRatioP)==1){mode=reading_global;}
       else if(sscanf(buf," RadiusRatioV %lf",&RadiusRatioV)==1){mode=reading_global;}
       else if(sscanf(buf," Density %lf %lf %lf %lf %lf %lf",&Density[0],&Density[1],&Density[2],&Density[3],&Density[4],&Density[5])==6){mode=reading_global;}
       else if(sscanf(buf," BulkModulus %lf %lf %lf %lf %lf %lf",&BulkModulus[0],&BulkModulus[1],&BulkModulus[2],&BulkModulus[3],&BulkModulus[4],&BulkModulus[5])==6){mode=reading_global;}
       else if(sscanf(buf," BulkViscosity %lf %lf %lf %lf %lf %lf",&BulkViscosity[0],&BulkViscosity[1],&BulkViscosity[2],&BulkViscosity[3],&BulkViscosity[4],&BulkViscosity[5])==6){mode=reading_global;}
       else if(sscanf(buf," ShearViscosity %lf %lf %lf %lf %lf %lf",&ShearViscosity[0],&ShearViscosity[1],&ShearViscosity[2],&ShearViscosity[3],&ShearViscosity[4],&ShearViscosity[5])==6){mode=reading_global;}
       else if(sscanf(buf," SurfaceTension %lf %lf %lf %lf",&SurfaceTension[0],&SurfaceTension[1],&SurfaceTension[4],&SurfaceTension[5])==4){mode=reading_global;}
       else if(sscanf(buf," YoungModulus %lf %lf %lf %lf",&YoungModulus[2],&YoungModulus[3],&YoungModulus[4],&YoungModulus[5])==4){mode=reading_global;}
       else if(sscanf(buf," PoissonRatio %lf %lf %lf %lf ",&PoissonRatio[2],&PoissonRatio[3],&PoissonRatio[4],&PoissonRatio[5])==4){mode=reading_global;}
       else if(sscanf(buf," Cohesion %lf %lf %lf %lf",&Cohesion[2],&Cohesion[3],&Cohesion[4],&Cohesion[5])==4){mode=reading_global;}
       else if(sscanf(buf," InternalFrictionAngle %lf %lf %lf %lf ",&InternalFrictionAngle[2],&InternalFrictionAngle[3],&InternalFrictionAngle[4],&InternalFrictionAngle[5])==4){mode=reading_global;}
       else if(sscanf(buf," DilatancyFrictionAngle %lf %lf %lf %lf ",&DilatancyFrictionAngle[2],&DilatancyFrictionAngle[3],&DilatancyFrictionAngle[4],&DilatancyFrictionAngle[5])==4){mode=reading_global;}
       else if(sscanf(buf," Enthalpy %lf %lf %lf %lf %lf %lf",&Heat[0],&Heat[1],&Heat[2],&Heat[3],&Heat[4],&Heat[5])==6){mode=reading_global;}
       else if(sscanf(buf," ThermalConductivity %lf %lf %lf %lf %lf %lf",&ThermalConductivity[0],&ThermalConductivity[1],&ThermalConductivity[2],&ThermalConductivity[3],&ThermalConductivity[4],&ThermalConductivity[5])==6){mode=reading_global;}
       else if(sscanf(buf," MeltingTemp %lf %lf %lf %lf %lf %lf",&MeltingPoint[0],&MeltingPoint[1],&MeltingPoint[2],&MeltingPoint[3],&MeltingPoint[4],&MeltingPoint[5])==6){mode=reading_global;}
       else if(sscanf(buf," SpecificHeat %lf %lf %lf %lf %lf %lf",&SpecificHeat[0],&SpecificHeat[1],&SpecificHeat[2],&SpecificHeat[3],&SpecificHeat[4],&SpecificHeat[5])==6){mode=reading_global;}
       else if(sscanf(buf," SolidifyingEnthalpy %lf %lf %lf %lf %lf %lf",&SolidifyingEnthalpy[0],&SolidifyingEnthalpy[1],&SolidifyingEnthalpy[2],&SolidifyingEnthalpy[3],&SolidifyingEnthalpy[4],&SolidifyingEnthalpy[5])==6){mode=reading_global;}
       else if(sscanf(buf," LiquefyingEnthalpy %lf %lf %lf %lf %lf %lf",&LiquefyingEnthalpy[0],&LiquefyingEnthalpy[1],&LiquefyingEnthalpy[2],&LiquefyingEnthalpy[3],&LiquefyingEnthalpy[4],&LiquefyingEnthalpy[5])==6){mode=reading_global;}
       else if(sscanf(buf," CriticalSolidFraction %lf %lf %lf %lf",&CriticalSolidFraction[0],&CriticalSolidFraction[1],&CriticalSolidFraction[2],&CriticalSolidFraction[3])==4){mode=reading_global;}
	   else if(sscanf(buf," InteractionRatio(Type0) %lf %lf %lf %lf %lf %lf",&InteractionRatio[0][0],&InteractionRatio[0][1],&InteractionRatio[0][2],&InteractionRatio[0][3],&InteractionRatio[0][4],&InteractionRatio[0][5])==6){mode=reading_global;}
       else if(sscanf(buf," InteractionRatio(Type1) %lf %lf %lf %lf %lf %lf",&InteractionRatio[1][0],&InteractionRatio[1][1],&InteractionRatio[1][2],&InteractionRatio[1][3],&InteractionRatio[1][4],&InteractionRatio[1][5])==6){mode=reading_global;}
       else if(sscanf(buf," InteractionRatio(Type2) %lf %lf %lf %lf %lf %lf",&InteractionRatio[2][0],&InteractionRatio[2][1],&InteractionRatio[2][2],&InteractionRatio[2][3],&InteractionRatio[2][4],&InteractionRatio[2][5])==6){mode=reading_global;}
       else if(sscanf(buf," InteractionRatio(Type3) %lf %lf %lf %lf %lf %lf",&InteractionRatio[3][0],&InteractionRatio[3][1],&InteractionRatio[3][2],&InteractionRatio[3][3],&InteractionRatio[3][4],&InteractionRatio[3][5])==6){mode=reading_global;}
       else if(sscanf(buf," InteractionRatio(Type4) %lf %lf %lf %lf %lf %lf",&InteractionRatio[4][0],&InteractionRatio[4][1],&InteractionRatio[4][2],&InteractionRatio[4][3],&InteractionRatio[4][4],&InteractionRatio[4][5])==6){mode=reading_global;}
       else if(sscanf(buf," InteractionRatio(Type5) %lf %lf %lf %lf %lf %lf",&InteractionRatio[5][0],&InteractionRatio[5][1],&InteractionRatio[5][2],&InteractionRatio[5][3],&InteractionRatio[5][4],&InteractionRatio[5][5])==6){mode=reading_global;}
       else if(sscanf(buf," Gravity %lf %lf %lf", &Gravity[0], &Gravity[1], &Gravity[2])==3){mode=reading_global;}
       else if(sscanf(buf," Wall2  Center %lf %lf %lf Velocity %lf %lf %lf Omega %lf %lf %lf", &WallCenter[4][0],  &WallCenter[4][1],  &WallCenter[4][2],  &WallVelocity[4][0],  &WallVelocity[4][1],  &WallVelocity[4][2],  &WallOmega[4][0],  &WallOmega[4][1],  &WallOmega[4][2])==9){mode=reading_global;}
       else if(sscanf(buf," Wall3  Center %lf %lf %lf Velocity %lf %lf %lf Omega %lf %lf %lf", &WallCenter[5][0],  &WallCenter[5][1],  &WallCenter[5][2],  &WallVelocity[5][0],  &WallVelocity[5][1],  &WallVelocity[5][2],  &WallOmega[5][0],  &WallOmega[5][1],  &WallOmega[5][2])==9){mode=reading_global;}
       else{
                log_printf("Invalid line in data file \"%s\"\n", buf);
            }
        }
    }
    fclose(fp);
	
	#pragma acc enter data create(ParticleCount,ParticleSpacing,ParticleVolume,Dt,Elastic_Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
	#pragma acc enter data create(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
	#pragma acc enter data create(Density[0:TYPE_COUNT],BulkModulus[0:TYPE_COUNT],BulkViscosity[0:TYPE_COUNT],ShearViscosity[0:TYPE_COUNT],SurfaceTension[0:TYPE_COUNT])
	#pragma acc enter data create(CofA[0:TYPE_COUNT],CofK,InteractionRatio[0:TYPE_COUNT][0:TYPE_COUNT])

	#pragma acc enter data create(Gravity[0:DIM])
//	#pragma acc enter data create(FluidParticleBegin,FluidParticleEnd,WallParticleBegin,WallParticleEnd)
	#pragma acc enter data create(PowerParticleCount,ParticleCountPower,CellWidth,CellCount[0:DIM])
	#pragma acc enter data create(WallCenter[0:WALL_END][0:DIM],WallVelocity[0:WALL_END][0:DIM],WallOmega[0:WALL_END][0:DIM])
	#pragma acc enter data create(WallRotation[0:WALL_END][0:DIM][0:DIM])
	#pragma acc enter data create(YoungModulus[0:TYPE_COUNT],PoissonRatio[0:TYPE_COUNT],Cohesion[0:TYPE_COUNT],InternalFrictionAngle[0:TYPE_COUNT],DilatancyFrictionAngle[0:TYPE_COUNT],CriticalSolidFraction[0:TYPE_COUNT])
	#pragma acc enter data create(Heat[0:TYPE_COUNT],ThermalConductivity[0:TYPE_COUNT],MeltingPoint[0:TYPE_COUNT],SpecificHeat[0:TYPE_COUNT],LiquefyingEnthalpy[0:TYPE_COUNT],SolidifyingEnthalpy[0:TYPE_COUNT])
	
}

static void readGridFile(char *filename)
{
    FILE *fp=fopen(filename,"r");
	char buf[1024];   
	
	
	try{
		
		if(fgets(buf,sizeof(buf),fp)==NULL)throw;
		sscanf(buf,"%lf",&Time);
		if(fgets(buf,sizeof(buf),fp)==NULL)throw;
		sscanf(buf,"%d  %lf  %lf %lf %lf  %lf %lf %lf",
			&ParticleCount,
			&ParticleSpacing,
			&DomainMin[0], &DomainMax[0],
			&DomainMin[1], &DomainMax[1],
			&DomainMin[2], &DomainMax[2]);
		#ifdef TWO_DIMENSIONAL
		ParticleVolume = ParticleSpacing*ParticleSpacing;
		#else
		ParticleVolume = ParticleSpacing*ParticleSpacing*ParticleSpacing;
		#endif
		
		Property = (int *)malloc(ParticleCount*sizeof(int));
        Position = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
        InitialPosition = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		Velocity = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		DensityA = (double *)malloc(ParticleCount*sizeof(double));
		GravityCenter = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		PressureA = (double *)malloc(ParticleCount*sizeof(double));
		VolStrainP = (double *)malloc(ParticleCount*sizeof(double));
		DivergenceP = (double *)malloc(ParticleCount*sizeof(double));
		PressureP = (double *)malloc(ParticleCount*sizeof(double));
		VirialPressureAtParticle = (double *)malloc(ParticleCount*sizeof(double));
		VirialStressAtParticle = (double (*) [DIM][DIM])malloc(ParticleCount*sizeof(double [DIM][DIM]));
		Mass = (double (*))malloc(ParticleCount*sizeof(double));
		Force = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		Mu = (double (*))malloc(ParticleCount*sizeof(double));
		Lambda = (double (*))malloc(ParticleCount*sizeof(double));
		Kappa = (double (*))malloc(ParticleCount*sizeof(double));
		
		#pragma acc enter data create(Property[0:ParticleCount])               attach(Property)
		#pragma acc enter data create(Position[0:ParticleCount][0:DIM])        attach(Position)
		#pragma acc enter data create(InitialPosition[0:ParticleCount][0:DIM]) attach(InitialPosition)
		#pragma acc enter data create(Velocity[0:ParticleCount][0:DIM])        attach(Velocity)
		#pragma acc enter data create(DensityA[0:ParticleCount])               attach(DensityA)
		#pragma acc enter data create(GravityCenter[0:ParticleCount][0:DIM])   attach(GravityCenter)
		#pragma acc enter data create(PressureA[0:ParticleCount])              attach(PressureA)
		#pragma acc enter data create(VolStrainP[0:ParticleCount])             attach(VolStrainP)
		#pragma acc enter data create(DivergenceP[0:ParticleCount])            attach(DivergenceP)
		#pragma acc enter data create(PressureP[0:ParticleCount])              attach(PressureP)
		#pragma acc enter data create(VirialPressureAtParticle[0:ParticleCount])               attach(VirialPressureAtParticle)
		#pragma acc enter data create(VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])   attach(VirialStressAtParticle)
		#pragma acc enter data create(Mass[0:ParticleCount])                attach(Mass)
		#pragma acc enter data create(Force[0:ParticleCount][0:DIM])        attach(Force)
		#pragma acc enter data create(Mu[0:ParticleCount])                  attach(Mu)
		#pragma acc enter data create(Lambda[0:ParticleCount])              attach(Lambda)
		#pragma acc enter data create(Kappa[0:ParticleCount])               attach(Kappa)
        
        Young =  (double (*))malloc(ParticleCount*sizeof(double));
      	LambdaLames = (double (*))malloc(ParticleCount*sizeof(double));
       	MuLames = (double (*))malloc(ParticleCount*sizeof(double));
        Strain = (double (*)[DIM][DIM])malloc(ParticleCount*sizeof(double [DIM][DIM]));
        Spin = (double (*)[DIM][DIM])malloc(ParticleCount*sizeof(double [DIM][DIM]));
        PlasticStrainRate = (double (*)[DIM][DIM])malloc(ParticleCount*sizeof(double [DIM][DIM]));
        Stress = (double (*)[DIM][DIM])malloc(ParticleCount*sizeof(double [DIM][DIM]));
        Acceleration = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
        DiffusiveCoefficient = (double (*))malloc(ParticleCount*sizeof(double));	
        ShearRate = (double (*))malloc(ParticleCount*sizeof(double));	

        #pragma acc enter data create(Young[0:ParticleCount]) attach(Young)
		#pragma acc enter data create(LambdaLames[0:ParticleCount]) attach(LambdaLames)
		#pragma acc enter data create(MuLames[0:ParticleCount]) attach(MuLames)
		#pragma acc enter data create(Strain[0:ParticleCount][0:DIM][0:DIM]) attach(Strain)
		#pragma acc enter data create(Spin[0:ParticleCount][0:DIM][0:DIM]) attach(Spin)
		#pragma acc enter data create(PlasticStrainRate[0:ParticleCount][0:DIM][0:DIM]) attach(PlasticStrainRate)
		#pragma acc enter data create(Stress[0:ParticleCount][0:DIM][0:DIM]) attach(Stress)
		#pragma acc enter data create(Acceleration[0:ParticleCount][0:DIM]) attach(Acceleration)
		#pragma acc enter data create(DiffusiveCoefficient[0:ParticleCount]) attach(DiffusiveCoefficient)
		#pragma acc enter data create(ShearRate[0:ParticleCount]) attach(ShearRate)

		// Parameters related to melting and solidification
		Temperature = (double *)malloc(ParticleCount * sizeof(double));
		Enthalpy = (double *)malloc(ParticleCount * sizeof(double));
		SolidFraction = (double *)malloc(ParticleCount * sizeof(double));
		Conductivity = (double *)malloc(ParticleCount * sizeof(double));
		MeltingTemp = (double *)malloc(ParticleCount * sizeof(double));
		Cp = (double *)malloc(ParticleCount * sizeof(double));
		H0 = (double *)malloc(ParticleCount * sizeof(double));
		H1 = (double *)malloc(ParticleCount * sizeof(double));
		#pragma acc enter data create(Temperature[0:ParticleCount]) attach(Temperature)
		#pragma acc enter data create(Enthalpy[0:ParticleCount]) attach(Enthalpy)
		#pragma acc enter data create(SolidFraction[0:ParticleCount]) attach(SolidFraction)
		#pragma acc enter data create(Conductivity[0:ParticleCount]) attach(Conductivity)
		#pragma acc enter data create(MeltingTemp[0:ParticleCount]) attach(MeltingTemp)
		#pragma acc enter data create(Cp[0:ParticleCount]) attach(Cp)
		#pragma acc enter data create(H0[0:ParticleCount]) attach(H0)
		#pragma acc enter data create(H1[0:ParticleCount]) attach(H1)
	


		NeighborCount  = (int *)malloc(ParticleCount*sizeof(int));
		Neighbor       = (int (*)[MAX_NEIGHBOR_COUNT])malloc(ParticleCount*sizeof(int [MAX_NEIGHBOR_COUNT]));
		NeighborCalculatedPosition = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		
		#pragma acc enter data create(NeighborCount[0:ParticleCount]) attach(NeighborCount)
		#pragma acc enter data create(Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT]) attach(Neighbor)
		#pragma acc enter data create(NeighborCalculatedPosition[0:ParticleCount][0:DIM]) attach(NeighborCalculatedPosition)
		
	//	double (*q)[DIM] = Position;
		double (*v)[DIM] = Velocity;
		
		for(int iP=0;iP<ParticleCount;++iP){
			if(fgets(buf,sizeof(buf),fp)==NULL)break;
			sscanf(buf,"%d  %lf %lf %lf %lf %lf %lf  %lf %lf %lf",
				&Property[iP],
				&Position[iP][0],&Position[iP][1],&Position[iP][2],
                &InitialPosition[iP][0],&InitialPosition[iP][1],&InitialPosition[iP][2],
				&v[iP][0],&v[iP][1],&v[iP][2]
			);
		}
	}catch(...){};
	
	fclose(fp);
	
    FluidParticleBegin = -1;
    FluidParticleEnd = -1;
    StructureParticleBegin = -1;
    StructureParticleEnd = -1;
    WallParticleBegin = -1;
    WallParticleEnd = -1;

    for (int iP = 0; iP < ParticleCount; ++iP) {
        int prop = Property[iP];

        if (FLUID_BEGIN <= prop && prop < FLUID_END) {
            if (FluidParticleBegin == -1) FluidParticleBegin = iP;
            FluidParticleEnd = iP + 1;
        } else if (STRUCTURE_BEGIN <= prop && prop < STRUCTURE_END) {
            if (StructureParticleBegin == -1) StructureParticleBegin = iP;
            StructureParticleEnd = iP + 1;
        } else if (WALL_BEGIN <= prop && prop < WALL_END) {
            if (WallParticleBegin == -1) WallParticleBegin = iP;
            WallParticleEnd = iP + 1;
        }
    }

    if (FluidParticleBegin != -1)
        printf("Fluid Particles: %d\n", FluidParticleEnd - FluidParticleBegin);
    else
        printf("Fluid Particles: 0\n");

    if (StructureParticleBegin != -1)
        printf("Structure Particles: %d\n", StructureParticleEnd - StructureParticleBegin);
    else
        printf("Structure Particles: 0\n");

    if (WallParticleBegin != -1)
        printf("Wall Particles: %d\n", WallParticleEnd - WallParticleBegin);
    else
        printf("Wall Particles: 0\n");

	#pragma acc update device(ParticleCount,ParticleSpacing,ParticleVolume,Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
	#pragma acc update device(Property[0:ParticleCount][0:DIM])
	#pragma acc update device(Position[0:ParticleCount][0:DIM])
	#pragma acc update device(InitialPosition[0:ParticleCount][0:DIM])
	#pragma acc update device(Velocity[0:ParticleCount][0:DIM])
//	#pragma acc update device(FluidParticleBegin,FluidParticleEnd,WallParticleBegin,WallParticleEnd)


	

	
}

static void writeProfFile(char *filename)
{
    FILE *fp=fopen(filename,"w");

    fprintf(fp,"%e\n",Time);
    fprintf(fp,"%d %e %e %e %e %e %e %e\n",
            ParticleCount,
            ParticleSpacing,
            DomainMin[0], DomainMax[0],
            DomainMin[1], DomainMax[1],
            DomainMin[2], DomainMax[2]);

 //   const double (*q)[DIM] = Position;
    const double (*v)[DIM] = Velocity;

    for(int iP=0;iP<ParticleCount;++iP){
            fprintf(fp,"%d %e %e %e %e %e %e  %e %e %e\n",
                    Property[iP],
                    Position[iP][0], Position[iP][1], Position[iP][2],
                    InitialPosition[iP][0],InitialPosition[iP][1],InitialPosition[iP][2],
                    v[iP][0], v[iP][1], v[iP][2]
            );
    }
    fflush(fp);
    fclose(fp);
}

static void writeVtkFile(char *filename)
{
	// update parameters to be output
	#pragma acc update host(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],InitialPosition[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],VirialPressureAtParticle[0:ParticleCount],Mass[0:ParticleCount],Temperature[0:ParticleCount],DiffusiveCoefficient[0:ParticleCount])
	#pragma acc update host(NeighborCount[0:ParticleCount],Force[0:ParticleCount][0:DIM],Young[0:ParticleCount])
	#pragma acc update host(Stress[0:ParticleCount][0:DIM][0:DIM],Strain[0:ParticleCount][0:DIM][0:DIM],PlasticStrainRate[0:ParticleCount][0:DIM][0:DIM],Spin[0:ParticleCount][0:DIM][0:DIM])
    const double (*v)[DIM] = Velocity;

    FILE *fp=fopen(filename, "w");

    fprintf(fp, "# vtk DataFile Version 2.0\n");
    fprintf(fp, "Unstructured Grid Example\n");
    fprintf(fp, "ASCII\n");

    fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
    fprintf(fp, "POINTS %d float\n", ParticleCount);
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "%e %e %e\n", (float)Position[iP][0], (float)Position[iP][1], (float)Position[iP][2]);
    }
    fprintf(fp, "CELLS %d %d\n", ParticleCount, 2*ParticleCount);
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "1 %d ",iP);
    }
    fprintf(fp, "\n");
    fprintf(fp, "CELL_TYPES %d\n", ParticleCount);
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "1 ");
    }
    fprintf(fp, "\n");

    fprintf(fp, "\n");

    fprintf(fp, "POINT_DATA %d\n", ParticleCount);
    fprintf(fp, "SCALARS label float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%d\n", Property[iP]);
    }
    fprintf(fp, "VECTORS velocity float\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e %e %e\n", (float)v[iP][0], (float)v[iP][1], (float)v[iP][2]);
    }
    fprintf(fp, "\n");

    /*
    fprintf(fp, "\n");
    fprintf(fp, "\n");
    fprintf(fp, "VECTORS displacement float\n");
    for(int iP=0;iP<ParticleCount;++iP){
        const double displacement[DIM]={Position[iP][0]-InitialPosition[iP][0],Position[iP][1]-InitialPosition[iP][1],Position[iP][2]-InitialPosition[iP][2]};
        fprintf(fp, "%e %e %e\n", (float)displacement[0], (float)displacement[1], (float)displacement[2]);
    }
    */
   
    for (int iD=0;iD<DIM;iD++){
       for(int jD=0;jD<DIM;jD++){
    fprintf(fp, "\n"); fprintf(fp," SCALARS stress[%d][%d] float \n", iD, jD);
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "%e\n", (float)Stress[iP][iD][jD]);
    }
    }
    }
        for (int iD=0;iD<DIM;iD++){
       for(int jD=0;jD<DIM;jD++){
    fprintf(fp, "\n"); fprintf(fp," SCALARS strain[%d][%d] float \n", iD, jD);
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "%e\n", (float)Strain[iP][iD][jD]);
    }
    }
    }
  //  for (int iD=0;iD<DIM;iD++){
 //  for(int jD=0;jD<DIM;jD++){
//fprintf(fp, "\n"); fprintf(fp," SCALARS spin[%d][%d] float \n", iD, jD);
//fprintf(fp, "LOOKUP_TABLE default\n");
//for(int iP=0;iP<ParticleCount;++iP){
//    fprintf(fp, "%e\n", (float)Spin[iP][iD][jD]);
//}
//}
//}

for (int iD=0;iD<DIM;iD++){
    for(int jD=0;jD<DIM;jD++){
 fprintf(fp, "\n"); fprintf(fp," SCALARS plastic[%d][%d] float \n", iD, jD);
 fprintf(fp, "LOOKUP_TABLE default\n");
 for(int iP=0;iP<ParticleCount;++iP){
     fprintf(fp, "%e\n", (float)PlasticStrainRate[iP][iD][jD]);
 }
 }
 }
    fprintf(fp, "SCALARS neighbor float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%d\n", NeighborCount[iP]);
    }
    fprintf(fp, "SCALARS Young float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e\n",(float) Young[iP]);
    }
	        fprintf(fp, "\n");
    fprintf(fp, "SCALARS Temperature float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "%e\n", (float)Temperature[iP]);
    }
   // fprintf(fp, "SCALARS Diffuse float 1\n");
   // fprintf(fp, "LOOKUP_TABLE default\n");
   // for(int iP=0;iP<ParticleCount;++iP){
   //      fprintf(fp, "%e\n",(float) DiffusiveCoefficient[iP]);
    //}
  //  fprintf(fp, "SCALARS shear float 1\n");
  //  fprintf(fp, "LOOKUP_TABLE default\n");
  //  for(int iP=0;iP<ParticleCount;++iP){
  //       fprintf(fp, "%e\n",(float) ShearRate[iP]);
  //  }

    fprintf(fp, "VECTORS force float\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e %e %e\n", (float)Force[iP][0], (float)Force[iP][1], (float)Force[iP][2]);
    }
    fprintf(fp, "\n");


    fprintf(fp, "SCALARS VirialPressureAtParticle float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e\n", (float)VirialPressureAtParticle[iP]); // trivial operation is done for 
    }
	fprintf(fp, "\n");
	
    fflush(fp);
    fclose(fp);
}

static void initializeWeight()
{
	RadiusRatioG = RadiusRatioA;
	
	RadiusA = RadiusRatioA*ParticleSpacing;
	RadiusG = RadiusRatioG*ParticleSpacing;
	RadiusP = RadiusRatioP*ParticleSpacing;
	RadiusV = RadiusRatioV*ParticleSpacing;
	
	
#ifdef TWO_DIMENSIONAL
		Swa = 1.0/2.0 * 2.0/15.0 * M_PI /ParticleSpacing/ParticleSpacing;
		Swg = 1.0/2.0 * 1.0/3.0 * M_PI /ParticleSpacing/ParticleSpacing;
		Swp = 1.0/2.0 * 1.0/3.0 * M_PI /ParticleSpacing/ParticleSpacing;
		Swv = 1.0/2.0 * 1.0/3.0 * M_PI /ParticleSpacing/ParticleSpacing;
		R2g = 1.0/2.0 * 1.0/30.0* M_PI *RadiusG*RadiusG /ParticleSpacing/ParticleSpacing /Swg;
#else	//code for three dimensional
		Swa = 1.0/3.0 * 1.0/5.0*M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
		Swg = 1.0/3.0 * 2.0/5.0 * M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
		Swp = 1.0/3.0 * 2.0/5.0 * M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
		Swv = 1.0/3.0 * 2.0/5.0 * M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
		R2g = 1.0/3.0 * 4.0/105.0*M_PI *RadiusG*RadiusG /ParticleSpacing/ParticleSpacing/ParticleSpacing /Swg;
#endif
	
	
	    {// N0a
        const double radius_ratio = RadiusA/ParticleSpacing;
        const int range = (int)(radius_ratio +3.0);
        int count = 0;
        double sum = 0.0;
#ifdef TWO_DIMENSIONAL
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                if(!(iX==0 && iY==0)){
                	const double x = ParticleSpacing * ((double)iX);
                	const double y = ParticleSpacing * ((double)iY);
                    const double rij2 = x*x + y*y;
                    if(rij2<=RadiusA*RadiusA){
                        const double rij = sqrt(rij2);
                        const double wij = wa(rij,RadiusA);
                        sum += wij;
                        count ++;
                    }
                }
            }
        }
#else	//code for three dimensional
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                for(int iZ=-range;iZ<=range;++iZ){
                    if(!(iX==0 && iY==0 && iZ==0)){
                    	const double x = ParticleSpacing * ((double)iX);
                    	const double y = ParticleSpacing * ((double)iY);
                    	const double z = ParticleSpacing * ((double)iZ);
                        const double rij2 = x*x + y*y + z*z;
                        if(rij2<=RadiusA*RadiusA){
                            const double rij = sqrt(rij2);
                            const double wij = wa(rij,RadiusA);
                            sum += wij;
                            count ++;
                        }
                    }
                }
            }
        }
#endif
        N0a = sum;
        log_printf("N0a = %e, count=%d\n", N0a, count);
    }	

    {// N0p
        const double radius_ratio = RadiusP/ParticleSpacing;
        const int range = (int)(radius_ratio +3.0);
        int count = 0;
        double sum = 0.0;
#ifdef TWO_DIMENSIONAL
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                if(!(iX==0 && iY==0)){
                	const double x = ParticleSpacing * ((double)iX);
                	const double y = ParticleSpacing * ((double)iY);
                    const double rij2 = x*x + y*y;
                    if(rij2<=RadiusP*RadiusP){
                        const double rij = sqrt(rij2);
                        const double wij = wp(rij,RadiusP);
                        sum += wij;
                        count ++;
                    }
                }
            }
        }
#else	//code for three dimensional
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                for(int iZ=-range;iZ<=range;++iZ){
                    if(!(iX==0 && iY==0 && iZ==0)){
                    	const double x = ParticleSpacing * ((double)iX);
                    	const double y = ParticleSpacing * ((double)iY);
                    	const double z = ParticleSpacing * ((double)iZ);
                        const double rij2 = x*x + y*y + z*z;
                        if(rij2<=RadiusP*RadiusP){
                            const double rij = sqrt(rij2);
                            const double wij = wp(rij,RadiusP);
                            sum += wij;
                            count ++;
                        }
                    }
                }
            }
        }
#endif
        N0p = sum;
        log_printf("N0p = %e, count=%d\n", N0p, count);
    }
	
	#pragma acc update device(RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
	

}


static void initializeFluid()
{
	for(int iP=0;iP<ParticleCount;++iP){
		Mass[iP]=Density[Property[iP]]*ParticleVolume;
	}
	for(int iP=0;iP<ParticleCount;++iP){
		Kappa[iP]=BulkModulus[Property[iP]];
	}
	for(int iP=0;iP<ParticleCount;++iP){
		Lambda[iP]=BulkViscosity[Property[iP]];
	}
	for(int iP=0;iP<ParticleCount;++iP){
		Mu[iP]=ShearViscosity[Property[iP]];
	}
    for(int iP=0;iP<ParticleCount;++iP){
        Enthalpy[iP]=Heat[Property[iP]];
    }
	   for(int iP=0;iP<ParticleCount;++iP){
        Young[iP]=YoungModulus[Property[iP]];
    }

	#ifdef TWO_DIMENSIONAL
	CofK = 0.350778153;
	double integN=0.024679383;
	double integX=0.226126699;
	#else 
	CofK = 0.326976006;
	double integN=0.021425779;
	double integX=0.233977488;
	#endif
	
	for(int iT=0;iT<TYPE_COUNT;++iT){
		CofA[iT]=SurfaceTension[iT] / ((RadiusG/ParticleSpacing)*(integN+CofK*CofK*integX));
	}
    
	#pragma acc update device(Mass[0:ParticleCount])
	#pragma acc update device(Kappa[0:ParticleCount])
	#pragma acc update device(Lambda[0:ParticleCount])
	#pragma acc update device(Mu[0:ParticleCount])
	#pragma acc update device(Young[0:ParticleCount])
	#pragma acc update device(CofK,CofA[0:TYPE_COUNT])

}



static void initializeWall()
{
	
	for(int iProp=WALL_BEGIN;iProp<WALL_END;++iProp){
		
		double theta;
		double normal[DIM]={0.0,0.0,0.0};
		double q[DIM+1];
		double t[DIM];
		double (&R)[DIM][DIM]=WallRotation[iProp];
		
		theta = abs(WallOmega[iProp][0]*WallOmega[iProp][0]+WallOmega[iProp][1]*WallOmega[iProp][1]+WallOmega[iProp][2]*WallOmega[iProp][2]);
		if(theta!=0.0){
			for(int iD=0;iD<DIM;++iD){
				normal[iD]=WallOmega[iProp][iD]/theta;
			}
		}
		q[0]=normal[0]*sin(theta*Dt/2.0);
		q[1]=normal[1]*sin(theta*Dt/2.0);
		q[2]=normal[2]*sin(theta*Dt/2.0);
		q[3]=cos(theta*Dt/2.0);
		t[0]=WallVelocity[iProp][0]*Dt;
		t[1]=WallVelocity[iProp][1]*Dt;
		t[2]=WallVelocity[iProp][2]*Dt;
		
		R[0][0] = q[0]*q[0]-q[1]*q[1]-q[2]*q[2]+q[3]*q[3];
		R[0][1] = 2.0*(q[0]*q[1]-q[2]*q[3]);
		R[0][2] = 2.0*(q[0]*q[2]+q[1]*q[3]);
		
		R[1][0] = 2.0*(q[0]*q[1]+q[2]*q[3]);
		R[1][1] = -q[0]*q[0]+q[1]*q[1]-q[2]*q[2]+q[3]*q[3];
		R[1][2] = 2.0*(q[1]*q[2]-q[0]*q[3]);
		
		R[2][0] = 2.0*(q[0]*q[2]-q[1]*q[3]);
		R[2][1] = 2.0*(q[1]*q[2]+q[0]*q[3]);
		R[2][2] = -q[0]*q[0]-q[1]*q[1]+q[2]*q[2]+q[3]*q[3];
		
	}
	#pragma acc update device(WallRotation[0:WALL_END][0:DIM][0:DIM])
}

static void initializeDomain( void )
{
	CellWidth = ParticleSpacing;
	
	double cellCount[DIM];
	
	cellCount[0] = round((DomainMax[0] - DomainMin[0])/CellWidth);
	cellCount[1] = round((DomainMax[1] - DomainMin[1])/CellWidth);
	#ifdef TWO_DIMENSIONAL
	cellCount[2] = 1;
	#else
	cellCount[2] = round((DomainMax[2] - DomainMin[2])/CellWidth);
	#endif
	
	CellCount[0] = (int)cellCount[0];
	CellCount[1] = (int)cellCount[1];
	CellCount[2] = (int)cellCount[2];
	CellCounts   = cellCount[0]*cellCount[1]*cellCount[2];
	
	if(cellCount[0]!=(double)CellCount[0] || cellCount[1]!=(double)CellCount[1] ||cellCount[2]!=(double)CellCount[2]){
		fprintf(stderr,"DomainWidth/CellWidth is not integer\n");
		DomainMax[0] = DomainMin[0] + CellWidth*(double)CellCount[0];
		DomainMax[1] = DomainMin[1] + CellWidth*(double)CellCount[1];
		DomainMax[2] = DomainMin[2] + CellWidth*(double)CellCount[2];
		fprintf(stderr,"Changing the Domain Max to (%e,%e,%e)\n", DomainMax[0], DomainMax[1], DomainMax[2]);
	}
	DomainWidth[0] = DomainMax[0] - DomainMin[0];
	DomainWidth[1] = DomainMax[1] - DomainMin[1];
	DomainWidth[2] = DomainMax[2] - DomainMin[2];
	
	CellParticleBegin = (int *)malloc( CellCounts * sizeof(int) );
	CellParticleEnd   = (int *)malloc( CellCounts * sizeof(int) );
	#pragma acc enter data create(CellParticleBegin[0:CellCounts]) attach(CellParticleBegin)
	#pragma acc enter data create(CellParticleEnd  [0:CellCounts]) attach(CellParticleEnd)
	
	
	// calculate minimun PowerParticleCount which sataisfies  ParticleCount < PowerParticleCount = pow(2,ParticleCountPower) 
	ParticleCountPower=0;  
	while((ParticleCount>>ParticleCountPower)!=0){
		++ParticleCountPower;
	}
	PowerParticleCount = (1<<ParticleCountPower);
	fprintf(stderr,"memory for CellIndex and CellParticle %d\n", PowerParticleCount );
	CellIndex    = (int *)malloc( (PowerParticleCount) * sizeof(int) );
	CellParticle = (int *)malloc( (PowerParticleCount) * sizeof(int) );
	#pragma acc enter data create(CellIndex   [0:PowerParticleCount]) attach(CellIndex)
	#pragma acc enter data create(CellParticle[0:PowerParticleCount]) attach(CellParticle)
	
	MaxRadius = ((RadiusA>MaxRadius) ? RadiusA : MaxRadius);
	MaxRadius = ((RadiusG>MaxRadius) ? RadiusG : MaxRadius);
	MaxRadius = ((RadiusP>MaxRadius) ? RadiusP : MaxRadius);
	MaxRadius = ((RadiusV>MaxRadius) ? RadiusV : MaxRadius);
	
	#pragma acc update device(CellWidth,CellCount[0:DIM],CellCounts)
	#pragma acc update device(DomainMax[0:DIM],DomainMin[0:DIM],DomainWidth[0:DIM])
	#pragma acc update device(ParticleCountPower,PowerParticleCount)
	#pragma acc update device(MaxRadius)
}


static int neighborCalculation( void ){
	double maxShift2=0.0;
	#pragma acc parallel loop reduction (max:maxShift2)
	#pragma omp parallel for reduction (max:maxShift2)
	for(int iP=0;iP<ParticleCount;++iP){
		 double disp[DIM];
         #pragma acc loop seq
         for(int iD=0;iD<DIM;++iD){
            disp[iD] = Mod(Position[iP][iD] - NeighborCalculatedPosition[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
         }
		const double shift2 = disp[0]*disp[0]+disp[1]*disp[1]+disp[2]*disp[2];
		if(shift2>maxShift2){
			maxShift2=shift2;
		}
	}
	
	if(maxShift2>0.5*MARGIN*0.5*MARGIN){
		return 1;
	}
	else{
		return 0;
	}
}




static void calculateNeighbor( void )
{
	
	// calculate CellIndex[iP]
	#pragma acc kernels present(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0; iP<(1<<ParticleCountPower); ++iP){
		if(iP<ParticleCount){
			const int iCX=((int)floor((Position[iP][0]-DomainMin[0])/CellWidth))%CellCount[0];
			const int iCY=((int)floor((Position[iP][1]-DomainMin[1])/CellWidth))%CellCount[1];
			const int iCZ=((int)floor((Position[iP][2]-DomainMin[2])/CellWidth))%CellCount[2];
			CellIndex[iP]=CellId(iCX,iCY,iCZ);
			CellParticle[iP]=iP;
		}
		else{
			CellIndex[ iP ]    = CellCount[0]*CellCount[1]*CellCount[2];
			CellParticle[ iP ] = ParticleCount;
		}
	}
	
	{
		// sort with CellIndex
		// https://edom18.hateblo.jp/entry/2020/09/21/150416
		for(int iMain=0;iMain<ParticleCountPower;++iMain){
			for(int iSub=0;iSub<=iMain;++iSub){
				
				int dist = (1<< (iMain-iSub));
				
				#pragma acc kernels present(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount])
				#pragma acc loop independent
				#pragma omp parallel for
				for(int iP=0;iP<(1<<ParticleCountPower);++iP){
					bool up = ((iP >> iMain) & 2) == 0;
					
					if(  (( iP & dist )==0) && ( CellIndex[ iP ] > CellIndex[ iP | dist ] == up) ){
						int tmpCellIndex    = CellIndex[ iP ];
						int tmpCellParticle = CellParticle[ iP ];
						CellIndex[ iP ]     = CellIndex[ iP | dist ];
						CellParticle[ iP ]  = CellParticle[ iP | dist ];
						CellIndex[ iP | dist ]    = tmpCellIndex;
						CellParticle[ iP | dist ] = tmpCellParticle;
					}
				}
			}
		}
	}
	
	// search for CellParticleBegin[iC]
	#pragma acc kernels present(CellParticleBegin[0:CellCounts],CellParticleEnd[0:CellCounts])
	{
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iC=0;iC<CellCounts;++iC){
			CellParticleBegin[iC]=0;
			CellParticleEnd[iC]=0;
		}
		
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iP=0; iP<ParticleCount; ++iP){
			if( CellIndex[iP]<CellIndex[iP+1] ){
				CellParticleEnd[ CellIndex[iP] ]   =iP+1;
				CellParticleBegin[ CellIndex[iP+1] ]=iP+1;
			}
		}
	}
    
    // calculate neighbor
	#pragma acc kernels present(Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],NeighborCount[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        NeighborCount[iP]=0;
    	for(int iN=0;iN<MAX_NEIGHBOR_COUNT;++iN){
    		Neighbor[iP][iN]=-1;
    	}
    }
	#pragma acc kernels present(CellParticle[0:PowerParticleCount],CellParticleBegin[0:CellCounts],CellParticleEnd[0:CellCounts],Position[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        const int range = (int)(ceil((MaxRadius+MARGIN)/CellWidth));
    	const int iCX=((int)floor((Position[iP][0]-DomainMin[0])/CellWidth))%CellCount[0];
    	const int iCY=((int)floor((Position[iP][1]-DomainMin[1])/CellWidth))%CellCount[1];
    	const int iCZ=((int)floor((Position[iP][2]-DomainMin[2])/CellWidth))%CellCount[2];

#ifdef TWO_DIMENSIONAL
    	#pragma acc loop seq
        for(int jCX=iCX-range;jCX<=iCX+range;++jCX){
        	#pragma acc loop seq
            for(int jCY=iCY-range;jCY<=iCY+range;++jCY){
                const int jCZ=0;
                const int jC=CellId(jCX,jCY,jCZ);
            	#pragma acc loop seq
                for(int jCP=CellParticleBegin[jC];jCP<CellParticleEnd[jC];++jCP){
                    const int jP=CellParticle[jCP];
                    double qij[DIM];
                	#pragma acc loop seq
                    for(int iD=0;iD<DIM;++iD){
                        qij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
                    }
                    const double qij2= qij[0]*qij[0]+qij[1]*qij[1]+qij[2]*qij[2];
                    if(qij2 <= (MaxRadius+MARGIN)*(MaxRadius+MARGIN)){
                        if(NeighborCount[iP]>=MAX_NEIGHBOR_COUNT){
                        	NeighborCount[iP]++;
                        }
                        else if(iP!=jP){
                            Neighbor[iP][NeighborCount[iP]] = jP;
                            NeighborCount[iP]++;
                        }
                    }
                }
            }
        }

    	    	
#else // TWO_DIMENSIONAL
    	#pragma acc loop seq
        for(int jCX=iCX-range;jCX<=iCX+range;++jCX){
        	#pragma acc loop seq
        	for(int jCY=iCY-range;jCY<=iCY+range;++jCY){
        		#pragma acc loop seq
                for(int jCZ=iCZ-range;jCZ<=iCZ+range;++jCZ){
                    const int jC=CellId(jCX,jCY,jCZ);
                	#pragma acc loop seq
                    for(int jCP=CellParticleBegin[jC];jCP<CellParticleEnd[jC];++jCP){
                        const int jP=CellParticle[jCP];
                        double qij[DIM];
                    	#pragma acc loop seq
                        for(int iD=0;iD<DIM;++iD){
                            qij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
                        }
                        const double qij2= qij[0]*qij[0]+qij[1]*qij[1]+qij[2]*qij[2];
                        if(qij2 <= (MaxRadius+MARGIN)*(MaxRadius+MARGIN)){
                            if(NeighborCount[iP]>=MAX_NEIGHBOR_COUNT){
                        		NeighborCount[iP]++;
                        	}
                        	else if(iP!=jP){
                            	Neighbor[iP][NeighborCount[iP]] = jP;
                            	NeighborCount[iP]++;
                        	}
                        }
                    }
                }
            }
        }
#endif // TWO_DIMENSIONAL
    }
	
	#pragma acc kernels present(NeighborCalculatedPosition[0:ParticleCount][0:DIM],Position[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			NeighborCalculatedPosition[iP][iD]=Position[iP][iD];
		}
	}
	
}

/*
// Function to select a free GPU and ensure simulation runs exclusively on it
static void selectFreeGPU() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA-capable devices found.\n");
        return;
    }

    int selectedDevice = -1;
    size_t maxFreeMem = 0;

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, device);

        if (err != cudaSuccess) {
            fprintf(stderr, "Error: Unable to get properties for device %d.\n", device);
            continue;
        }

        size_t freeMem = 0, totalMem = 0;
        cudaSetDevice(device);
        err = cudaMemGetInfo(&freeMem, &totalMem);

        if (err != cudaSuccess) {
            fprintf(stderr, "Error: Unable to get memory info for device %d.\n", device);
            continue;
        }

        printf("Device %d: %s, Free Memory: %.2f MB, Total Memory: %.2f MB\n",
               device, prop.name, freeMem / (1024.0 * 1024.0), totalMem / (1024.0 * 1024.0));

        // Update selected device if this one has more free memory
        if (freeMem > maxFreeMem) {
            maxFreeMem = freeMem;
            selectedDevice = device;
        }
    }

    if (selectedDevice != -1) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, selectedDevice);
        printf("Selecting GPU %d (%s) with %.2f MB free memory.\n", 
               selectedDevice, prop.name, maxFreeMem / (1024.0 * 1024.0));
        acc_set_device_num(selectedDevice, acc_device_nvidia); // Set for OpenACC
        cudaSetDevice(selectedDevice);                         // Set for CUDA

        // Ensure the GPU is exclusively used
        err = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error: Unable to set exclusive GPU mode for device %d.\n", selectedDevice);
        }
    } else {
        printf("No suitable GPU found with sufficient memory. Using default GPU.\n");
    }
}
*/




static void calculateConvection()
{
#pragma acc kernels
#pragma acc loop independent
#pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){

        Acceleration[iP][0] += Force[iP][0]/Mass[iP];
        Acceleration[iP][1] += Force[iP][1]/Mass[iP];
        Acceleration[iP][2] += Force[iP][2]/Mass[iP];

        Position[iP][0] += Velocity[iP][0]*Dt;
        Position[iP][1] += Velocity[iP][1]*Dt;
        Position[iP][2] += Velocity[iP][2]*Dt;
    }
}


static void resetForce()
{
	#pragma acc kernels present(Force[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
    	#pragma acc loop seq
        for(int iD=0;iD<DIM;++iD){
            Force[iP][iD]=0.0;
        }
    }
}


static void calculatePhysicalCoefficients()
{	
  #pragma acc kernels present (Property[0:ParticleCount],Mass[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        Mass[iP]=Density[Property[iP]]*ParticleVolume;
    }
    
    #pragma acc kernels present (Kappa[0:ParticleCount],Property[0:ParticleCount],VolStrainP[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        Kappa[iP]=BulkModulus[Property[iP]];
        if(VolStrainP[iP]<0.0){Kappa[iP]=0.0;}
    }
   
    
    #pragma acc kernels present(Lambda[0:ParticleCount],VolStrainP[0:ParticleCount],Property[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
  Lambda[iP]=BulkViscosity[Property[iP]];
        if(VolStrainP[iP]<0.0){Lambda[iP]=0.0;}
    }
    
    #pragma acc kernels present (Property[0:ParticleCount],Mu[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        Mu[iP]=ShearViscosity[Property[iP]];
    }
	#pragma acc kernels present (Property[0:ParticleCount],Young[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        Young[iP]=YoungModulus[Property[iP]];
    }
	    #pragma acc kernels present (Property[0:ParticleCount],Conductivity[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        Conductivity[iP]=ThermalConductivity[Property[iP]];
    }
    #pragma acc kernels present (Property[0:ParticleCount],MeltingTemp[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        MeltingTemp[iP]=MeltingPoint[Property[iP]];
    }
    #pragma acc kernels present (Property[0:ParticleCount],Cp[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        Cp[iP]=SpecificHeat[Property[iP]];
    }
    #pragma acc kernels present (Property[0:ParticleCount],H0[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        H0[iP]=SolidifyingEnthalpy[Property[iP]];
    }
    #pragma acc kernels present (Property[0:ParticleCount],H1[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        H1[iP]=LiquefyingEnthalpy[Property[iP]];
    }

}


static void calculateDensityA()
{
    
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],DensityA[0:ParticleCount],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT])
	{	
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iP=0;iP<ParticleCount;++iP){
            if(STRUCTURE_BEGIN<=Property[iP] && Property[iP]<STRUCTURE_END ) continue;
			double sum = 0.0;
			#pragma acc loop seq
			for(int iN=0;iN<NeighborCount[iP];++iN){
				const int jP=Neighbor[iP][iN];
				double ratio = InteractionRatio[Property[iP]][Property[jP]];
				double xij[DIM];
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
				}
				const double radius = RadiusA;
				const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
				if(radius*radius - rij2 >= 0){
					const double rij = sqrt(rij2);
					const double weight = ratio * wa(rij,radius);
					sum += weight;
				}
			}
			DensityA[iP]=sum;
		}
	}
}


static void calculateGravityCenter()
{
 

	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],GravityCenter[0:ParticleCount][0:DIM])
	{
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iP=0;iP<ParticleCount;++iP){
            if(STRUCTURE_BEGIN<=Property[iP] && Property[iP]<STRUCTURE_END ) continue;
			double sum[DIM]={0.0,0.0,0.0};
			#pragma acc loop seq
			for(int iN=0;iN<NeighborCount[iP];++iN){
				const int jP=Neighbor[iP][iN];
				double ratio = InteractionRatio[Property[iP]][Property[jP]];
				double xij[DIM];
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
				}
				const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
				if(RadiusG*RadiusG - rij2 >= 0){
					const double rij = sqrt(rij2);
					const double weight = ratio * wg(rij,RadiusG);
					#pragma acc loop seq
					for(int iD=0;iD<DIM;++iD){
						sum[iD] += xij[iD]*weight/R2g*RadiusG;
					}
				}
			}
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				GravityCenter[iP][iD] = sum[iD];
			}
		}
	}
}

static void calculatePressureA()
{

	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],DensityA[0:ParticleCount],PressureA[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		PressureA[iP] = CofA[Property[iP]]*(DensityA[iP]-N0a)/ParticleSpacing;
		if(N0a<=DensityA[iP]){
			PressureA[iP] = 0.0;
		}
	}
	
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],PressureA[0:ParticleCount],DensityA[0:ParticleCount],PressureA[0:ParticleCount],Force[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
       if(STRUCTURE_BEGIN<=Property[iP] && Property[iP]<STRUCTURE_END ) continue; 
    	double force[DIM]={0.0,0.0,0.0};
    	#pragma acc loop seq
        for(int iN=0;iN<NeighborCount[iP];++iN){
            const int jP=Neighbor[iP][iN];
			double ratio_ij = InteractionRatio[Property[iP]][Property[jP]];
        	double ratio_ji = InteractionRatio[Property[jP]][Property[iP]];
            double xij[DIM];
        	#pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
            const double radius = RadiusA;
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            if(radius*radius - rij2 > 0){
                const double rij = sqrt(rij2);
                const double dwij = ratio_ij * dwadr(rij,radius);
            	const double dwji = ratio_ji * dwadr(rij,radius);
                const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
            	#pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    force[iD] += (PressureA[iP]*dwij+PressureA[jP]*dwji)*eij[iD]* ParticleVolume;
                }
            }
        }
    	#pragma acc loop seq
    	for(int iD=0;iD<DIM;++iD){
    		Force[iP][iD] += force[iD];
    	}
    }
}

static void calculateDiffuseInterface()
{
	
	
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],GravityCenter[0:ParticleCount][0:DIM],Force[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
        if(STRUCTURE_BEGIN<=Property[iP] && Property[iP]<STRUCTURE_END ) continue;
		const double ai = CofA[Property[iP]]*(CofK)*(CofK);
		double force[DIM]={0.0,0.0,0.0};
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
			const int jP=Neighbor[iP][iN];
			const double aj = CofA[Property[iP]]*(CofK)*(CofK);
			double ratio_ij = InteractionRatio[Property[iP]][Property[jP]];
			double ratio_ji = InteractionRatio[Property[jP]][Property[iP]];
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if(RadiusG*RadiusG - rij2 > 0){
				const double rij = sqrt(rij2);
				const double wij = ratio_ij * wg(rij,RadiusG);
				const double wji = ratio_ji * wg(rij,RadiusG);
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					force[iD] -= (aj*GravityCenter[jP][iD]*wji-ai*GravityCenter[iP][iD]*wij)/R2g*RadiusG * (ParticleVolume/ParticleSpacing);
				}
				const double dwij = ratio_ij * dwgdr(rij,RadiusG);
				const double dwji = ratio_ji * dwgdr(rij,RadiusG);
				const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
				double gr=0.0;
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					gr += (aj*GravityCenter[jP][iD]*dwji-ai*GravityCenter[iP][iD]*dwij)*xij[iD];
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					force[iD] -= (gr)*eij[iD]/R2g*RadiusG * (ParticleVolume/ParticleSpacing);
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			Force[iP][iD]+=force[iD];
		}
	}
}

static void calculateDensityP()
{
	
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],VolStrainP[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double sum = 0.0;
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
			const int jP=Neighbor[iP][iN];
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			const double radius = RadiusP;
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if(radius*radius - rij2 >= 0){
				const double rij = sqrt(rij2);
				const double weight = wp(rij,radius);
				sum += weight;
			}
		}
		VolStrainP[iP] = (sum - N0p);
	}
}

static void calculateDivergenceP()
{

	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],Velocity[0:ParticleCount][0:DIM],DivergenceP[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double sum = 0.0;
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
            const int jP=Neighbor[iP][iN];
            double xij[DIM];
			#pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
			const double radius = RadiusP;
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if(radius*radius - rij2 >= 0){
				const double rij = sqrt(rij2);
				const double dw = dwpdr(rij,radius);
				double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
				double uij[DIM];
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					uij[iD]=Velocity[jP][iD]-Velocity[iP][iD];
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					sum -= uij[iD]*eij[iD]*dw;
				}
			}
		}
		DivergenceP[iP]=sum;
	}
}

static void calculatePressureP()
{
	
	#pragma acc kernels present (PressureP[0:ParticleCount],Lambda[0:ParticleCount],DivergenceP[0:ParticleCount],VolStrainP[0:ParticleCount],Kappa[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		PressureP[iP] = -Lambda[iP]*DivergenceP[iP];
		if(VolStrainP[iP]>0.0){
			PressureP[iP]+=Kappa[iP]*VolStrainP[iP];
		}
	}
	
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],PressureP[0:ParticleCount],Force[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
	if(STRUCTURE_BEGIN<=Property[iP] && Property[iP]<STRUCTURE_END ) continue;
		double force[DIM]={0.0,0.0,0.0};
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
            const int jP=Neighbor[iP][iN];
            double xij[DIM];
			#pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
			const double radius = RadiusP;
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if(radius*radius - rij2 > 0){
			
				const double rij = sqrt(rij2);
				const double dw = dwpdr(rij,radius);
				double gradw[DIM] = {dw*xij[0]/rij,dw*xij[1]/rij,dw*xij[2]/rij};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					force[iD] += (PressureP[iP]+PressureP[jP])*gradw[iD]*ParticleVolume;
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			Force[iP][iD]+=force[iD];
		}
	}
}



static void calculateViscosityV(){

	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],Velocity[0:ParticleCount][0:DIM],Mu[0:ParticleCount],Force[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
    	double force[DIM]={0.0,0.0,0.0};
    	 if(STRUCTURE_BEGIN<=Property[iP] && Property[iP]<STRUCTURE_END ) continue;
    	#pragma acc loop seq
        for(int iN=0;iN<NeighborCount[iP];++iN){
            const int jP=Neighbor[iP][iN];
            double xij[DIM];
        	#pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            
        	if(RadiusV*RadiusV - rij2 > 0){
                const double rij = sqrt(rij2);
                const double dwij = -dwvdr(rij,RadiusV);
            	const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
        		double uij[DIM];
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					uij[iD]=Velocity[jP][iD]-Velocity[iP][iD];
				}
				const double muij = 2.0*(Mu[iP]*Mu[jP])/(Mu[iP]+Mu[jP]);
            	double fij[DIM] = {0.0,0.0,0.0};
        		#pragma acc loop seq
            	for(int iD=0;iD<DIM;++iD){
            		#ifdef TWO_DIMENSIONAL
            		force[iD] += 8.0*muij*(uij[0]*eij[0]+uij[1]*eij[1]+uij[2]*eij[2])*eij[iD]*dwij/rij*ParticleVolume;
            		#else
            		force[iD] += 10.0*muij*(uij[0]*eij[0]+uij[1]*eij[1]+uij[2]*eij[2])*eij[iD]*dwij/rij*ParticleVolume;
            		#endif
            	}
            }
        }
    	#pragma acc loop seq
    	for(int iD=0;iD<DIM;++iD){
    		Force[iP][iD] += force[iD];
    	}
    }
}

//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//=================Energy Conservation calculation======================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//






static void calculateEnergyConservation(){
    
#pragma acc kernels present(Temperature[0:ParticleCount],MeltingTemp[0:ParticleCount],Enthalpy[0:ParticleCount],H0[0:ParticleCount],H1[0:ParticleCount],Cp[0:ParticleCount])
#pragma acc loop independent
#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;iP++){
        if(Enthalpy[iP]<H0[iP]){
            Temperature[iP]=MeltingTemp[iP]+(Enthalpy[iP]-H0[iP])/Cp[iP];
        }
        else if(H0[iP]<=Enthalpy[iP] && Enthalpy[iP]<H1[iP]){
            Temperature[iP] =MeltingTemp[iP];
            
        }
        else if(Enthalpy[iP]>=H1[iP]) {
            Temperature[iP] = MeltingTemp[iP]+ (Enthalpy[iP]-H1[iP])/Cp[iP];
            
        }
    }
    
#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Temperature[0:ParticleCount],Conductivity[0:ParticleCount],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],Enthalpy[0:ParticleCount])
#pragma acc loop independent
#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        double flux=0.0;
       #pragma acc loop seq
            for(int iN=0;iN<NeighborCount[iP];++iN){
                const int jP=Neighbor[iP][iN];
            if(iP==jP)continue;
            double xij[DIM];
            #pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
}
            
            const double radius = RadiusV;
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            if(radius*radius - rij2 >= 0){
                const double rij = sqrt(rij2);
                const double dwij = -dwvdr(rij,radius);
                const double wwij = dwij/rij;
                flux += 2.0*Conductivity[iP]/(Density[Property[iP]])*(Temperature[jP]-Temperature[iP])*wwij*Dt;
            
            }
        }
        Enthalpy[iP] += flux;

}

#define emissivity 0.8
#define SBC 5.67e-8

#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Temperature[0:ParticleCount],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],Enthalpy[0:ParticleCount])
#pragma acc loop independent
#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        const double TE = 5000;
        double flux=0.0;
        #pragma acc loop seq
        for(int iN=0;iN<NeighborCount[iP];++iN){
            const int jP=Neighbor[iP][iN];
            if(iP==jP)continue;
            double xij[DIM];
            #pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
}
            
            const double radius = RadiusV;
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
        if(radius*radius - rij2 >= 0){
			if( NeighborCount[iP] <15){
          double T4 = Temperature[iP] * Temperature[iP] * Temperature[iP] * Temperature[iP];
          double Te4 = TE * TE * TE * TE;

        flux += SBC * emissivity / Mass[iP] * ParticleSpacing * ParticleSpacing *(Te4)*Dt;
            }
    }
        }
        Enthalpy[iP] += flux;
    }

}



static void calculateSolidFraction(){
    #pragma acc kernels present(Enthalpy[0:ParticleCount],H0[0:ParticleCount],H1[0:ParticleCount],SolidFraction[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;iP++){
        if(Enthalpy[iP]<H0[iP]){
            SolidFraction[iP] = 1.0;
        }
        else if(H0[iP]<=Enthalpy[iP] && Enthalpy[iP]< H1[iP]){
            SolidFraction[iP] = (H1[iP]-Enthalpy[iP])/(H1[iP]-H0[iP]);
        }
        else if(Enthalpy[iP]>=H1[iP]) {
            SolidFraction[iP] = 0.0;
        }
    }
      #pragma acc kernels present(SolidFraction[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=WallParticleBegin;iP<WallParticleEnd;iP++){
        SolidFraction[iP] = 0.0;
    }
}

static void calculateViscosity(){
    
    
 #pragma acc kernels present(SolidFraction[0:ParticleCount],Mu[0:ParticleCount],Property[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;iP++){
        if(SolidFraction[iP]<CriticalSolidFraction[Property[iP]]){
            LambdaLames[iP] = ShearViscosity[Property[iP]]*exp(2.5*3.0*SolidFraction[iP]);
        }
        else {
            Mu[iP] = 1000*ShearViscosity[Property[iP]]*exp(2.5*3.0*SolidFraction[iP]);
        }
    }
    

     #pragma acc kernels present(Mu[0:ParticleCount],Property[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=StructureParticleBegin;iP<StructureParticleEnd;iP++){
     //   Young[iP] = Young[iP]*exp(0.0023/Temperature[iP]);
	//	 MuLames[iP] =  MuLames[iP]*exp(0.0023/Temperature[iP]);
    }
}


//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//=================Elastoplastic calculation============================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//


static void calculateLamesconstant()
{
#pragma acc kernels present(Property[0:ParticleCount],MuLames[0:ParticleCount],LambdaLames[0:ParticleCount],Young[0:ParticleCount])
#pragma acc loop independent
#pragma omp parallel for
	 for(int iP=0;iP<ParticleCount;++iP){
    const double E = Young[iP];
    const double v = PoissonRatio[Property[iP]];

        LambdaLames[iP]=(E*v)/((1+v)*(1-2*v));
        MuLames[iP]=E/(2*(1+v));
    }
}


static void resetAcceleration()
{
	#pragma acc kernels loop present(Acceleration[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
    #pragma acc loop seq
        for (int iD = 0; iD < DIM; ++iD) {
            Acceleration[iP][iD]=0.0;
        }
	}
}

static void calculateStrainRateTensor() {
	#pragma acc parallel loop present(Position[0:ParticleCount][0:DIM], \
									  Velocity[0:ParticleCount][0:DIM], \
									  Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT], \
									  NeighborCount[0:ParticleCount], \
									  Strain[0:ParticleCount][0:DIM][0:DIM], \
									  ShearRate[0:ParticleCount], \
									  DomainWidth[0:DIM])
 #pragma omp parallel for
 for(int iP=0;iP<ParticleCount;++iP){
		double eps[3][3] = {{0.0}};
	#pragma acc loop seq
        for(int iN=0;iN<NeighborCount[iP];++iN){
            const int jP=Neighbor[iP][iN];
			if(iP==jP) continue;
            double xij[DIM];
        	#pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }

            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if (rij2 <= RadiusP * RadiusP){

			const double rij = sqrt(rij2);
			const double dw = dwpdr(rij, RadiusP);
			const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};

			double uij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
			uij[iD]=Velocity[jP][iD]-Velocity[iP][iD];
			}

			#pragma acc loop seq
			for (int iD = 0; iD < DIM; ++iD)
				#pragma acc loop seq
				for (int jD = 0; jD < DIM; ++jD)
					eps[iD][jD] += 0.5 * (uij[iD] * eij[jD] + uij[jD] * eij[iD]) * dw;
		}
	}
		double ss = 0.0;
		#pragma acc loop seq
		for (int iD = 0; iD < DIM; ++iD)
			#pragma acc loop seq
			for (int jD = 0; jD < DIM; ++jD) {
				Strain[iP][iD][jD] = eps[iD][jD];
				ss += eps[iD][jD] * eps[iD][jD];
		}
		ShearRate[iP] = sqrt(0.5 * ss);
	}
}




static void calculateSpinTensor() {
	#pragma acc parallel loop present(Position[0:ParticleCount][0:DIM], \
									  Velocity[0:ParticleCount][0:DIM], \
									  Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT], \
									  NeighborCount[0:ParticleCount], \
									  DomainWidth[0:DIM], \
									  Spin[0:ParticleCount][0:DIM][0:DIM])
	#pragma omp parallel for	  
       for(int iP=0;iP<ParticleCount;++iP){
		double omega[DIM][DIM] = {{0.0}};
#pragma acc loop seq
		for (int iN = 0; iN < NeighborCount[iP]; ++iN) {
			const int jP = Neighbor[iP][iN];
			if (iP == jP) continue;

			double xij[DIM];

			#pragma acc loop seq
			for (int iD = 0; iD < DIM; ++iD) {
				xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] + 0.5 * DomainWidth[iD], DomainWidth[iD]) - 0.5 * DomainWidth[iD];			
			}

            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if (rij2 <= RadiusP * RadiusP){
			const double rij = sqrt(rij2);
			const double dw = dwpdr(rij, RadiusP);
			const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};

			double uij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
			uij[iD]=Velocity[jP][iD]-Velocity[iP][iD];
			}

			#pragma acc loop seq
			for (int iD = 0; iD < DIM; ++iD)
				#pragma acc loop seq
				for (int jD = 0; jD < DIM; ++jD)
					omega[iD][jD] += 0.5 * (uij[iD] * eij[jD] - uij[jD] * eij[iD]) * dw;
		}
	}

		#pragma acc loop seq
		for (int iD = 0; iD < DIM; ++iD)
			#pragma acc loop seq
			for (int jD = 0; jD < DIM; ++jD)
				Spin[iP][iD][jD] = omega[iD][jD];
	}
}


static void calculatePlasticStrainRateTensor() {

	#pragma acc parallel loop present(Strain[0:ParticleCount][0:DIM][0:DIM], \
									  Stress[0:ParticleCount][0:DIM][0:DIM], \
									  PlasticStrainRate[0:ParticleCount][0:DIM][0:DIM], \
									  MuLames[0:ParticleCount], \
									  LambdaLames[0:ParticleCount], \
									  InternalFrictionAngle[0:TYPE_COUNT], \
									  DilatancyFrictionAngle[0:TYPE_COUNT], \
									  Cohesion[0:TYPE_COUNT], \
									  Property[0:ParticleCount],DiffusiveCoefficient[0:ParticleCount],ShearRate[0:ParticleCount])
	 #pragma omp parallel for
      for(int iP=0;iP<ParticleCount;++iP){
		double eps_dot[DIM][DIM] = {{0}}, sigma[DIM][DIM] = {{0}}, s[DIM][DIM] = {{0}};
		double trace_eps_dot = 0.0, I1 = 0.0;

		// Load tensors and compute trace of strain rate and stress
		#pragma acc loop seq
		for (int iD = 0; iD < DIM; ++iD) {
			#pragma acc loop seq
			for (int jD = 0; jD < DIM; ++jD) {
				eps_dot[iD][jD] = Strain[iP][iD][jD];
				sigma[iD][jD] = Stress[iP][iD][jD];
				if (iD == jD) {
					trace_eps_dot += eps_dot[iD][jD];
					I1 += sigma[iD][jD];
				}
			}
		}
		const double p = I1 / DIM;
		double I[DIM][DIM] = {{1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0}};

		// Deviatoric stress and J2
		double J2 = 0.0;
		#pragma acc loop seq
		for (int iD = 0; iD <DIM; ++iD) {
			#pragma acc loop seq
			for (int jD = 0; jD < DIM; ++jD) {
			//	double delta = (iD == jD) ? 1.0 : 0.0;
				s[iD][jD] = sigma[iD][jD] - p *I[iD][jD];
				J2 += 0.5 * s[iD][jD] * s[iD][jD];
			}
		}

		double q = sqrt(fmax(3.0 * J2, 1e-12)); // prevent division by zero


		// Drucker–Prager parameters
		double phi_deg = InternalFrictionAngle[Property[iP]];
		double psi_deg = DilatancyFrictionAngle[Property[iP]];
		double cohesion = Cohesion[Property[iP]];
		double phi = phi_deg * M_PI / 180.0;
		double psi = psi_deg * M_PI / 180.0;

		double tan_phi = tan(phi);
		double tan_psi = tan(psi);
		double alpha_phi = (3.0 * tan_phi) / sqrt(9.0 + 12.0 * tan_phi * tan_phi);
		double alpha_psi = (3.0 * tan_psi) / sqrt(9.0 + 12.0 * tan_psi * tan_psi);
		double k = 3.0 * cohesion / sqrt(9.0 + 12.0 * tan_phi * tan_phi);

		// Yield function and plastic potential
		double f = q - alpha_phi * p - k;
		double g = q - alpha_psi * p;

		if(ShearRate[iP] == 0.0) continue;
		DiffusiveCoefficient[iP] = alpha_phi*p/(2*ShearRate[iP]);
		
		
		if (f <= 1e-8) {
			#pragma acc loop seq
			for (int iD = 0; iD < DIM; ++iD)
			#pragma acc loop seq
				for (int jD = 0; jD < DIM; ++jD)
					PlasticStrainRate[iP][iD][jD] = 0.0;
			continue;
		}

		// Compute flow directions
		double df[DIM][DIM] = {{0}}, dg[DIM][DIM] = {{0}};
		double df_dot_eps = 0.0, df_dot_dg = 0.0;
		#pragma acc loop seq
		for (int iD = 0; iD <DIM; ++iD) {
			#pragma acc loop seq
			for (int jD = 0; jD < DIM; ++jD) {
				df[iD][jD] = (3.0 / (2.0 * q)) * s[iD][jD] - alpha_phi/DIM * I[iD][jD];
				dg[iD][jD] = (3.0 / (2.0 * q)) * s[iD][jD] - alpha_psi/DIM * I[iD][jD];
				df_dot_eps += df[iD][jD] * eps_dot[iD][jD];
				df_dot_dg  += df[iD][jD] * dg[iD][jD];
			}
		}

		// Compute plastic multiplier
		double mu = MuLames[iP];
		double lambda = LambdaLames[iP];
		double tr_df = -alpha_phi;
		double tr_dg = -alpha_psi;
		double num = lambda * tr_df * trace_eps_dot + 2.0 * mu * df_dot_eps;
		double den = lambda * tr_df * tr_dg + 2.0 * mu * df_dot_dg;
// Ensure denominator is numerically stable
if (fabs(den) < 1e-12) {
    den = (den >= 0.0) ? 1e-12 : -1e-12;
}
	double dLambda = num / den;

	//	}
		if (dLambda < 0.0) dLambda = 0.0;

		// Set symmetric plastic strain rate
		#pragma acc loop seq
		for (int iD = 0; iD < DIM; ++iD) {
			#pragma acc loop seq
			for (int jD = 0; jD < DIM; ++jD) {
		PlasticStrainRate[iP][iD][jD] =  dLambda * dg[iD][jD];

			}
		}
	}
}

	
				
    static void calculateDiffusive() {
        #pragma acc kernels present(Position[0:ParticleCount][0:DIM], \
                                    Velocity[0:ParticleCount][0:DIM], \
                                    Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT], \
                                    NeighborCount[0:ParticleCount], \
                                    DomainWidth[0:DIM], \
                                    Mu[0:ParticleCount], \
                                    ParticleVolume, \
                                    Force[0:ParticleCount][0:DIM])
        #pragma acc loop independent
        #pragma omp parallel for
     for(int iP=0;iP<ParticleCount;++iP){
            double force[DIM] = {0.0};
			#pragma acc loop seq
            for (int iN = 0; iN < NeighborCount[iP]; ++iN) {
                const int jP = Neighbor[iP][iN];
                if (iP == jP) continue;
    
                double xij[DIM];
				#pragma acc loop seq
                for (int iD = 0; iD < DIM; ++iD) {
                    xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] + 0.5 * DomainWidth[iD], DomainWidth[iD]) - 0.5 * DomainWidth[iD];
                   
				}
                const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
                if (rij2 <= RadiusV * RadiusV) {
    
                const double rij = sqrt(rij2);
                const double dw = -dwvdr(rij, RadiusV);
				const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};

				double  uij[DIM];
				#pragma acc loop seq
                for (int iD = 0; iD < DIM; ++iD) {
                    uij[iD] = Velocity[jP][iD] - Velocity[iP][iD];
                }

				#pragma acc loop seq
                for (int iD = 0; iD < DIM; ++iD) {
                    force[iD] += 2.0*Mu[iP]* uij[iD] * dw/rij * ParticleVolume;
                }
            }
		}
			#pragma acc loop seq
            for (int iD = 0; iD < DIM; ++iD) {
                Force[iP][iD] += force[iD];
            }
        }
    }

// -----------------------------------------------------------------------------
// Stress update with elastoplastic strain rate and Jaumann correction
// -----------------------------------------------------------------------------
static void calculateStress() {
	#pragma acc parallel loop present(Stress[0:ParticleCount][0:DIM][0:DIM], \
									  Strain[0:ParticleCount][0:DIM][0:DIM], \
									  PlasticStrainRate[0:ParticleCount][0:DIM][0:DIM], \
									  Spin[0:ParticleCount][0:DIM][0:DIM], \
									  MuLames[0:ParticleCount], \
									  LambdaLames[0:ParticleCount])
	#pragma omp parallel for
	for (int iP = 0; iP < ParticleCount; ++iP) {

		// Local tensors
		double eps_dot[DIM][DIM]     = {{0.0}}; // total strain rate
		double eps_dot_p[DIM][DIM]   = {{0.0}}; // plastic strain rate
		double omega_dot[DIM][DIM]   = {{0.0}}; // spin tensor
		double sigma_dot[DIM][DIM]   = {{0.0}}; // stress rate (elastic-plastic)
		double omega_sigma[DIM][DIM] = {{0.0}}; // ω·σ term
		double sigma_omega[DIM][DIM] = {{0.0}}; // σ·ω term
		double jaumann_term[DIM][DIM]= {{0.0}}; // total stress increment
		double tr_eps = 0.0, tr_eps_p = 0.0;    // volumetric strain rates

		// ---------------------------------------------------------------------
		// Load strain rate, plastic strain rate, and spin tensor
		// ---------------------------------------------------------------------
		#pragma acc loop seq
		for (int iD = 0; iD < DIM; ++iD) {
			#pragma acc loop seq
			for (int jD = 0; jD < DIM; ++jD) {
				eps_dot[iD][jD]   = Strain[iP][iD][jD];
				eps_dot_p[iD][jD] = PlasticStrainRate[iP][iD][jD];
				omega_dot[iD][jD] = Spin[iP][iD][jD];
				if (iD == jD) {
					tr_eps   += eps_dot[iD][jD];
					tr_eps_p += eps_dot_p[iD][jD];
				}
			}
		}

		// ---------------------------------------------------------------------
		// Material constants
		// ---------------------------------------------------------------------
		double mu     = MuLames[iP];
		double lambda = LambdaLames[iP];

		// ---------------------------------------------------------------------
		// Elastic–plastic stress rate (Hooke’s law with plastic strain subtraction)
		// σ̇ = λ (tr(ε̇) - tr(ε̇ᵖ)) δ + 2μ (ε̇ - ε̇ᵖ)
		// ---------------------------------------------------------------------
		#pragma acc loop seq
		for (int iD = 0; iD < DIM; ++iD) {
			#pragma acc loop seq
			for (int jD = 0; jD < DIM; ++jD) {
				double delta_ij = (iD == jD) ? 1.0 : 0.0;
				double elastic_strain = eps_dot[iD][jD] - eps_dot_p[iD][jD];
				sigma_dot[iD][jD] = lambda * (tr_eps - tr_eps_p) * delta_ij
				                  + 2.0 * mu * elastic_strain;
			}
		}

		// ---------------------------------------------------------------------
		// Jaumann objective stress rate correction
		// σ̇* = σ̇ + σ·ω - ω·σ
		// ---------------------------------------------------------------------
		#pragma acc loop seq
		for (int iD = 0; iD < DIM; ++iD) {
			#pragma acc loop seq
			for (int jD = 0; jD < DIM; ++jD) {
				#pragma acc loop seq
				for (int kD = 0; kD < DIM; ++kD) {
					omega_sigma[iD][jD] += omega_dot[iD][kD] * Stress[iP][kD][jD];
					sigma_omega[iD][jD] += Stress[iP][iD][kD] * omega_dot[kD][jD];
				}
			}
		}

		// ---------------------------------------------------------------------
		// Final stress update with time increment Δt
		// σⁿ⁺¹ = σⁿ + Δt (σ̇ + σ·ω - ω·σ)
		// ---------------------------------------------------------------------
		#pragma acc loop seq
		for (int iD = 0; iD < DIM; ++iD) {
			#pragma acc loop seq
			for (int jD = 0; jD < DIM; ++jD) {
				jaumann_term[iD][jD] = sigma_dot[iD][jD]
				                     + sigma_omega[iD][jD]
				                     - omega_sigma[iD][jD];
				Stress[iP][iD][jD] += Elastic_Dt * jaumann_term[iD][jD];
			}
		}
	}
}



static void calculateStressForce() {

	#pragma acc parallel loop present(Position[0:ParticleCount][0:DIM], \
	                                  Stress[0:ParticleCount][0:DIM][0:DIM], \
	                                  Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT], \
	                                  NeighborCount[0:ParticleCount], \
	                                  DomainWidth[0:DIM], \
	                                  Force[0:ParticleCount][0:DIM], \
	                                  Property[0:ParticleCount]) // ParticleVolume, RadiusP visible globally?
	#pragma omp parallel for
	for (int iP = 0; iP < ParticleCount; ++iP) {

		double fi[DIM] = {0.0, 0.0, 0.0};

		#pragma acc loop seq
		for (int n = 0; n < NeighborCount[iP]; ++n) {
			const int jP = Neighbor[iP][n];
			if (iP == jP) continue; // 保険（通常は隣接リストに自分は入らない）

			// minimal image displacement x_j - x_i (periodic)
			double xij[DIM];
			#pragma acc loop seq
			for (int d = 0; d < DIM; ++d) {
				xij[d] = Mod(Position[jP][d] - Position[iP][d] + 0.5 * DomainWidth[d], DomainWidth[d]) - 0.5 * DomainWidth[d];
			}

			const double rij2 = xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2];
			if (rij2 > RadiusP * RadiusP) continue;

			// avoid division by zero
			if (rij2 < 1e-20) continue;

			const double rij  = sqrt(rij2);
			const double dWdr = dwpdr(rij, RadiusP);

			// unit vector and kernel gradient
			const double eij[DIM] = { xij[0]/rij, xij[1]/rij, xij[2]/rij };
			// ∇W_ij = eij * dWdr

			// symmetric stress tensor sum S_ij = σ_i + σ_j
			double Se[DIM] = {0.0, 0.0, 0.0}; // vector = (σ_i + σ_j) · eij
			#pragma acc loop seq
			for (int a = 0; a < DIM; ++a) {
				double dot = 0.0;
				#pragma acc loop seq
				for (int b = 0; b < DIM; ++b) {
					dot += (Stress[iP][a][b] + Stress[jP][a][b]) * eij[b];
				}
				Se[a] = dot;
			}

			// If particle volumes vary, consider V_j instead of global ParticleVolume
			const double Vj = ParticleVolume; // replace with per-particle volume if available

			#pragma acc loop seq
			for (int a = 0; a < DIM; ++a) {
				fi[a] += Se[a] * dWdr * Vj;
			}
			// Note: No minus sign here because we discretize ∇·σ directly.
			// If you assemble momentum as ρ du/dt = ∇·σ + ρg, this sign is consistent.
			// If you instead sum "internal force" as −∇·σ, insert a leading minus.
		}

		#pragma acc loop seq
		for (int a = 0; a < DIM; ++a) {
			Force[iP][a] += fi[a];
		}
	}
}


		

static void updateElasticForce()
{

#pragma acc kernels
#pragma acc loop independent
#pragma omp parallel for
    for(int iP=StructureParticleBegin;iP<StructureParticleEnd;++iP){
            Velocity[iP][0] += Force[iP][0]/Mass[iP]*Elastic_Dt;
            Velocity[iP][1] += Force[iP][1]/Mass[iP]*Elastic_Dt;
            Velocity[iP][2] += Force[iP][2]/Mass[iP]*Elastic_Dt;
    }
}


static void updateElasticPosition()
{

#pragma acc kernels
#pragma acc loop independent
#pragma omp parallel for
    for(int iP=StructureParticleBegin;iP<StructureParticleEnd;++iP){
            Position[iP][0] += Velocity[iP][0]*Elastic_Dt;
            Position[iP][1] += Velocity[iP][1]*Elastic_Dt;
            Position[iP][2] += Velocity[iP][2]*Elastic_Dt;
    
    }
}

//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//=================Elastoplastic calculation============================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//
//======================================================================//




static void calculateGravity(){
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
        Force[iP][0] += Mass[iP]*Gravity[0];
        Force[iP][1] += Mass[iP]*Gravity[1];
        Force[iP][2] += Mass[iP]*Gravity[2];
    }
    
    #pragma acc kernels
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=StructureParticleBegin;iP<StructureParticleEnd;++iP){
	//	if(Position[iP][1]<0.1){
    Force[iP][0] += Mass[iP]*Gravity[0];
    Force[iP][1] += Mass[iP]*Gravity[1];
    Force[iP][2] += Mass[iP]*Gravity[2];
	//	}
    }
}

static void calculateAcceleration()
{
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
        Velocity[iP][0] += Force[iP][0]/Mass[iP]*Dt;
        Velocity[iP][1] += Force[iP][1]/Mass[iP]*Dt;
        Velocity[iP][2] += Force[iP][2]/Mass[iP]*Dt;
    }
}


static void calculateWall()
{
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=WallParticleBegin;iP<WallParticleEnd;++iP){
        Force[iP][0] = 0.0;
        Force[iP][1] = 0.0;
        Force[iP][2] = 0.0;
    }
	  if (Time > 0.30 && Time < 0.40) {
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=WallParticleBegin;iP<WallParticleEnd;++iP){
		const int iProp = Property[iP];
		double r[DIM] = {Position[iP][0]-WallCenter[iProp][0],Position[iP][1]-WallCenter[iProp][1],Position[iP][2]-WallCenter[iProp][2]};
		const double (&R)[DIM][DIM] = WallRotation[iProp];
		const double (&w)[DIM] = WallOmega[iProp];
		r[0] = R[0][0]*r[0]+R[0][1]*r[1]+R[0][2]*r[2];
		r[1] = R[1][0]*r[0]+R[1][1]*r[1]+R[1][2]*r[2];
		r[2] = R[2][0]*r[0]+R[2][1]*r[1]+R[2][2]*r[2];
		Velocity[iP][0] = w[1]*r[2]-w[2]*r[1] + WallVelocity[iProp][0];
		Velocity[iP][1] = w[2]*r[0]-w[0]*r[2] + WallVelocity[iProp][1];
		Velocity[iP][2] = w[0]*r[1]-w[1]*r[0] + WallVelocity[iProp][2];
		Position[iP][0] = r[0] + WallCenter[iProp][0] + WallVelocity[iProp][0]*Dt;
		Position[iP][1] = r[1] + WallCenter[iProp][1] + WallVelocity[iProp][1]*Dt;
		Position[iP][2] = r[2] + WallCenter[iProp][2] + WallVelocity[iProp][2]*Dt;
		
	}
	}
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iProp=WALL_BEGIN;iProp<WALL_END;++iProp){
		WallCenter[iProp][0] += WallVelocity[iProp][0]*Dt;
		WallCenter[iProp][1] += WallVelocity[iProp][1]*Dt;
		WallCenter[iProp][2] += WallVelocity[iProp][2]*Dt;
	}
	
}




static void calculateVirialStressAtParticle()
{
	//const double (*x)[DIM] = Position;
	const double (*v)[DIM] = Velocity;
	

	#pragma acc kernels present (VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD]=0.0;
			}
		}
	}
	
	#pragma acc kernels present(Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double stress[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
			const int jP=Neighbor[iP][iN];
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			
			// pressureP
			if(RadiusP*RadiusP - rij2 > 0){
				const double rij = sqrt(rij2);
				const double dwij = dwpdr(rij,RadiusP);
				double gradw[DIM] = {dwij*xij[0]/rij,dwij*xij[1]/rij,dwij*xij[2]/rij};
				double fij[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					fij[iD] = (PressureP[iP])*gradw[iD]*ParticleVolume;
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#pragma acc loop seq
					for(int jD=0;jD<DIM;++jD){
						stress[iD][jD]+=1.0*fij[iD]*xij[jD]/ParticleVolume;
					}
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD] += stress[iD][jD];
			}
		}
	}
	
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc loop independent	
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double stress[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
			const int jP=Neighbor[iP][iN];
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			
			
			// pressureA
			if(RadiusA*RadiusA - rij2 > 0){
				double ratio = InteractionRatio[Property[iP]][Property[jP]];
				const double rij = sqrt(rij2);
				const double dwij = ratio * dwadr(rij,RadiusA);
				double gradw[DIM] = {dwij*xij[0]/rij,dwij*xij[1]/rij,dwij*xij[2]/rij};
				double fij[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					fij[iD] = (PressureA[iP])*gradw[iD]*ParticleVolume;
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#pragma acc loop seq
					for(int jD=0;jD<DIM;++jD){
						stress[iD][jD]+=1.0*fij[iD]*xij[jD]/ParticleVolume;
					}
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD] += stress[iD][jD];
			}
		}

	}
	
	#pragma acc kernels present(Position[0:ParticleCount][0:DIM],v[0:ParticleCount][0:DIM],Mu[0:ParticleCount],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc loop independent	
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double stress[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
			const int jP=Neighbor[iP][iN];
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			
			
			// viscosity term
			if(RadiusV*RadiusV - rij2 > 0){
				const double rij = sqrt(rij2);
				const double dwij = -dwvdr(rij,RadiusV);
				const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
				const double vij[DIM] = {v[jP][0]-v[iP][0],v[jP][1]-v[iP][1],v[jP][2]-v[iP][2]};
				const double muij = 2.0*(Mu[iP]*Mu[jP])/(Mu[iP]+Mu[jP]);
				double fij[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#ifdef TWO_DIMENSIONAL
					fij[iD] = 8.0*muij*(vij[0]*eij[0]+vij[1]*eij[1]+vij[2]*eij[2])*eij[iD]*dwij/rij*ParticleVolume;
					#else
					fij[iD] = 10.0*muij*(vij[0]*eij[0]+vij[1]*eij[1]+vij[2]*eij[2])*eij[iD]*dwij/rij*ParticleVolume;
					#endif
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#pragma acc loop seq
					for(int jD=0;jD<DIM;++jD){
						stress[iD][jD]+=0.5*fij[iD]*xij[jD]/ParticleVolume;
					}
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD] += stress[iD][jD];
			}
		}
	}
	
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc loop independent	
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double stress[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
			const int jP=Neighbor[iP][iN];
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			
			
			// diffuse interface force (1st term)
			if(RadiusG*RadiusG - rij2 > 0){
				const double a = CofA[Property[iP]]*(CofK)*(CofK);
				double ratio = InteractionRatio[Property[iP]][Property[jP]];
				const double rij = sqrt(rij2);
				const double weight = ratio * wg(rij,RadiusG);
				double fij[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					fij[iD] = -a*( -GravityCenter[iP][iD])*weight/R2g*RadiusG * (ParticleVolume/ParticleSpacing);
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#pragma acc loop seq
					for(int jD=0;jD<DIM;++jD){
						stress[iD][jD]+=1.0*fij[iD]*xij[jD]/ParticleVolume;
					}
				}
			}
			
			// diffuse interface force (2nd term)
			if(RadiusG*RadiusG - rij2 > 0.0){
				const double a = CofA[Property[iP]]*(CofK)*(CofK);
				double ratio = InteractionRatio[Property[iP]][Property[jP]];
				const double rij = sqrt(rij2);
				const double dw = ratio * dwgdr(rij,RadiusG);
				const double gradw[DIM] = {dw*xij[0]/rij,dw*xij[1]/rij,dw*xij[2]/rij};
				double gr=0.0;
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					gr += (                     -GravityCenter[iP][iD])*xij[iD];
				}
				double fij[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					fij[iD] = -a*(gr)*gradw[iD]/R2g*RadiusG * (ParticleVolume/ParticleSpacing);
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#pragma acc loop seq
					for(int jD=0;jD<DIM;++jD){
						stress[iD][jD]+=1.0*fij[iD]*xij[jD]/ParticleVolume;
					}
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD] += stress[iD][jD];
			}
		}
	}	
	

	#pragma acc kernels present(VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM],VirialPressureAtParticle[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#ifdef TWO_DIMENSIONAL
		VirialPressureAtParticle[iP]=-1.0/2.0*(VirialStressAtParticle[iP][0][0]+VirialStressAtParticle[iP][1][1]);
		#else 
		VirialPressureAtParticle[iP]=-1.0/3.0*(VirialStressAtParticle[iP][0][0]+VirialStressAtParticle[iP][1][1]+VirialStressAtParticle[iP][2][2]);
		#endif
	}

}



static void calculatePeriodicBoundary( void )
{
	#pragma acc kernels present(Position[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
    	#pragma acc loop seq
        for(int iD=0;iD<DIM;++iD){
            Position[iP][iD] = Mod(Position[iP][iD]-DomainMin[iD],DomainWidth[iD])+DomainMin[iD];
        }
    }
}

