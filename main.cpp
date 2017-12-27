// TestOcv.cpp : Defines the entry point for the console application.
//

#include "DetectDefect.h"

int main(int argc, char** argv)
{
	DetectDefect dd = DetectDefect();
	ReturnValue rv = dd.GetDefect(argv[1]);
	int k = 0;
	k++;
	return 0;
}
