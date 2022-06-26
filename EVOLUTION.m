%% core functions of level set methods
function u = EVOLUTION(u0, g, lambda, mu, nu, epsilon, delt, numIter)
u=u0;
[vx,vy]=gradient(g);
 
for k=1:numIter
    u=NeumannBoundCond(u);
    [ux,uy]=gradient(u); 
    normDu=sqrt(ux.^2 + uy.^2 + 1e-10);
    Nx=ux./normDu;
    Ny=uy./normDu;
    diracU=Dirac(u,epsilon);
    K=curvature_central(Nx,Ny);
    weightedLengthTerm=lambda*diracU.*(vx.*Nx + vy.*Ny + g.*K);
    penalizingTerm=mu*(4*del2(u)-K);
    weightedAreaTerm=nu.*diracU.*g;
    u=u+delt*(weightedLengthTerm + weightedAreaTerm + penalizingTerm); 
    % update the level set function
end