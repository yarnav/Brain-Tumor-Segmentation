function K = curvature_central(nx,ny)
[nxx,junk]=gradient(nx);  
[junk,nyy]=gradient(ny);
K=nxx+nyy;