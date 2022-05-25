% close all
% clc

clear all
profile on

addpath('img'); 
addpath('util');
%%%%%%%%%%%%%%  Data parameters


num_iter = 300;
inner_loop = 1;
threshold_res = 2e-3; %1e-3

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
scale = 5;

lambda = 90;  %100 %160 %90
p1 = 1; %  %1%0.1
p2 = 2; %  %2%0.5
%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%  original image
uclean = double(imread('peppers.png'));
 
%%%%%%%%%%%%%%  noisy image
noise_std = 20; %20 %30
u0 = uclean + noise_std*randn(size(uclean));
    

%%%%%%%%%%%%%%  Initialization
[l1, l2] = size(u0);
Area = l1*l2;

lambda11 = u0*0; lambda12 = u0*0; 
lambda21 = u0*0; lambda22 = u0*0; lambda23 = u0*0;

x1 = u0*0; x2 = u0*0; 
y1 = u0*0; y2 = u0*0; y3 = u0*0; 
u = u0*0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  z1p = -1 + exp( sqrt(-1)*2*pi*[1:l1]/l1);
%  z1n =  1 - exp(-sqrt(-1)*2*pi*[1:l1]/l1);    
%  z2p = -1 + exp( sqrt(-1)*2*pi*[1:l2]/l2);
%  z2n =  1 - exp(-sqrt(-1)*2*pi*[1:l2]/l2);
% 
 A = zeros(l1,l2);
 scale1 = sqrt(scale);
 scale2 = scale^2;
 for i=1:l1
     for j=1:l2
%         
        A(i,j) = -2*(p1/scale2)*( cos(2*pi*i/l1)+cos(2*pi*j/l2)-2 ) + 4*(p2/scale2)*( cos(2*pi*i/l1)+cos(2*pi*j/l2)-2 ).^2  + 1/lambda;
% 
    end
 end

% DtD = abs(psf2otf([1,-1],[l1, l2])).^2 + abs(psf2otf([1;-1],[l1, l2])).^2;
% A = (p1/scale2)*DtD + (p2/scale2)*(DtD.^2) + 1/lambda; 
 

residual = zeros(2,num_iter);
energy = zeros(1,num_iter);
relative_error = zeros(1,num_iter);
error = zeros(2,num_iter);

energy(1) = ZC_energyMC(u,u0,x1,x2,y1,y2,y3,0,0,lambda,scale);
residual(1:2,1) = 0;
relative_error(1) = 0;
real_iter = num_iter;
record_flag = 0;

t = cputime;

for iter = 2:num_iter

for LL = 1 : inner_loop
    
    %%%%%%%%%%%%%%%%%%%%%  For u
    
    u_old = u;
    
    g = u0./lambda - dxb(p1*x1-lambda11,scale) - dyb(p1*x2-lambda12,scale) + dxxb(p2*y1-lambda21,scale1) + 2*dxyb(p2*y2-lambda22,scale1) + dyyb(p2*y3-lambda23,scale1);
    g = fftn(g);
    u = real(ifftn(g./A));
    
    %%%%%%%%%%%%%%%%%%%%%  For x
    ux = dxf(u,scale);
    uy = dyf(u,scale);
    bta = 1./ sqrt(1 + ux.^2 + uy.^2);
    afax = dxf(bta,scale);
    afay = dyf(bta,scale);
    afa = sqrt(afax.^2 + afay.^2);
    
    xx = dxf(u,scale) + lambda11/p1;
    xy = dyf(u,scale) + lambda12/p1;
    x = sqrt(xx.^2 + xy.^2);
    x(x==0) = 1;
    x = max(x - afa/p1,0)./x;
    x1 = xx.*x;
    x2 = xy.*x;

    %%%%%%%%%%%%%%%%%%%%%  For y
  
    yx = dxxf(u,scale1) + lambda21/p2;
    yy = dxyf(u,scale1) + lambda22/p2;
    yz = dyyf(u,scale1) + lambda23/p2;
    y = sqrt(yx.^2 + 2*yy.^2 + yz.^2);
    y(y==0) = 1;
    y = max(y - bta/p2,0)./y;
    y1 = yx.*y;
    y2 = yy.*y;
    y3 = yz.*y;
    
end    

    %%%%%%%%%%%%%%%%%%%%%  For Lambda
    lambda11_old = lambda11;
    lambda12_old = lambda12;
    lambda21_old = lambda21;
    lambda22_old = lambda22;
    lambda23_old = lambda23;
    
    lambda11 = lambda11 - p1*(x1 - dxf(u,scale));
    lambda12 = lambda12 - p1*(x2 - dyf(u,scale));
 
    lambda21 = lambda21 - p2*(y1 - dxxf(u,scale1));
    lambda22 = lambda22 - p2*(y2 - dxyf(u,scale1));
    lambda23 = lambda23 - p2*(y3 - dyyf(u,scale1));

    %%%%%%%%%%%%%%%%%%%%%  For residual
    
    R11 = x1 - dxf(u,scale);
    R12 = x2 - dyf(u,scale);
    
    R21 = y1 - dxxf(u,scale1);
    R22 = y2 - dxyf(u,scale1);
    R23 = y3 - dyyf(u,scale1);
    
    residual(1,iter) = sum( abs(R11(:))+abs(R12(:)) )/Area;
    residual(2,iter) = sum( abs(R21(:))+2*abs(R22(:))+abs(R23(:)) )/Area;

    %%%%%%%%%%%%%%%%%%%%%  For energy    
    energy(iter) = ZC_energyMC(u,u0,x1,x2,y1,y2,y3,afa,bta,lambda,scale);

    %%%%%%%%%%%%%%%%%%%%%  For convergence
   
   error(1,iter) = sum(sum(abs(lambda11-lambda11_old)+abs(lambda12-lambda12_old)))/Area; 
   error(2,iter) = sum(sum(abs(lambda21-lambda21_old)+2*abs(lambda22-lambda22_old)+abs(lambda23-lambda23_old)))/Area; 
    relative_error(iter) = sum(sum( abs(u-u_old) ))/Area;   
    if( relative_error(iter) < threshold_res )
        if( record_flag==0 )
            real_iter = iter;
            record_flag = 1;
        end
    end

end
t = cputime - t;

%%%%%%%%%%%%%%%%%% saving data......................

save u0.dat u0 -ASCII;
save u.dat  u  -ASCII;

fprintf(' The iteration number is: %10d\n', real_iter);
fprintf(' The iteration time is: %4.2fs', t);
fprintf(' The relative error is: %10.8f\n', relative_error(real_iter));

psnr_u = psnr(uint8(u),uint8(uclean))
ssim_u = ssim(uint8(u),uint8(uclean))

%%%%%%%%%%%%%%%%%% showing results..................

iternum = 1:iter;
figure;
plot(iternum,log(relative_error),'r','LineWidth',2); %/2
xlabel('Iteration')
ylabel('Relative error in u^k')
%title('relative error');
set(gca,'FontWeight','bold')

figure;
plot(iternum,log(residual(1,:)),'r',iternum,log(residual(2,:)),'g','LineWidth',2); %/2
xlabel('Iteration')
ylabel('Relative residuals')
legend('R1','R2','Location','NorthEast')
%title('The relative residuals');
set(gca,'FontWeight','bold')

figure;
plot(iternum,log(error(1,:)),'r',iternum,log(error(2,:)),'g','LineWidth',2);  % /2
xlabel('Iteration')
ylabel('Relative errors in multipliers')
legend('L1','L2','Location','NorthEast')
%title('The error in multipliers');
set(gca,'FontWeight','bold')

% xx1 = 1:l1;
% figure;
% plot(xx1,u0(60,:),'r',xx1,u(60,:),'g',xx1,uclean(60,:),'b','LineWidth',1.5);
% set(gca,'FontWeight','bold')

figure;
plot(iternum,log(energy),'b','LineWidth',2);
xlabel('Iteration')
ylabel('Energy')
%title('The energy');
set(gca,'FontWeight','bold')

figure;
imshow(uint8(u0));
%title('The input noisy image');

figure;
imshow(uint8(u));
%title('The result');

figure;
imshow(afa,[]);colorbar

figure;
imshow(bta,[]);colorbar

figure;
imshow(uint8(u0-u+100));
%title('The difference');

profile off
profile viewer
