
function energy = ZC_energyMC(u,u0,x1,x2,y1,y2,y3,afa,bta,lambda,h)

%%%%%%%%%%%%%%%%%%%%%  Energy
norm_x = sqrt(x1.^2 + x2.^2);
normx = afa.*norm_x;
norm_y = sqrt(y1.^2 + 2*(y2.^2) + y3.^2);
normy = bta.*norm_y;
energy = (0.5/lambda)*norm(u-u0,'fro')^2 + (sum(normx(:)) + sum(normy(:)));
%energy =  (sum(normx(:)) + sum(normy(:)));
energy = energy*h*h;

