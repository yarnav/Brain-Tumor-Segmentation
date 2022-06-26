%% Read Test Image

clear all
close all
clc


[filename,pathname] = uigetfile('*.jpg;*.tif;*.png;*.jpeg;*.bmp;*.pgm;*.gif','pick an imgae');
file = fullfile(pathname,filename);

   img = imread(file);
   figure,imshow(img);
   title('Test Image');
   
%% Preprocessing - Image Enhancement
  

if size(img,3) == 3
    
    img=rgb2gray(img);
    
end

figure,imshow(img);
title('Gray Image');

EI=imadjust(img);
figure,
imshow(EI);
title('Enhanced Image');

%% Feature Extraction using GLCM

g = graycomatrix(EI);
stats = graycoprops(g,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;

FV=[Contrast Correlation Energy Homogeneity];

%% Generate Template based on reduce gray level intensity

TI = uint8(floor(double(EI)/8));
figure,imshow(TI);
title('Template Image with Gray Level 32');

%% Coarse Image 

im=bitxor(EI,TI);
figure,imshow(uint8(im));
title('Coarse Image');

%% Segmentation using K-means

no_of_cluster=4;
varargin=[];

[m,n,p] = size(im);

    gray = uint8(im);
    
    minimum = min(gray(:));
    vector=double((gray(:)-minimum)+1);% 1
    vector = repmat(vector,[1,no_of_cluster]);
    vec_mean=(1:no_of_cluster).*max((vector))/(no_of_cluster+1);
    num = length(vector);
    itr = 0;
   
    %================ for gray image ==========================
    while(true)
        itr = itr+1;
        old_mean=vec_mean;
        vec_mean = repmat(vec_mean,[num,1]);
        %     for i=1:length(label_vector)
        distance=(((vector-vec_mean)).^2);
        vec_mean(2:end,:)=[];
        [~,label_vector] = min(distance,[],2);
        % i=1:no_of_cluster;
        for i=1:no_of_cluster
            index=(label_vector==i);
            vec_mean(:,i)=sum(vector(index))/nnz(index);
        end
        if (vec_mean==old_mean | itr>105)% You can change it accordingly
            break;
        end
    end
    label_im = reshape(label_vector,size(gray));
    figure,imshow(label_im,[]);
    title('Segmented Image using K-means');
    
 %% Segmentation using Modified Fuzzy C-means
 
  ncluster=5;
  
  imgg=label_im,[];
    
  expo=2;

  max_iter=100;
  
img=wiener2(imgg,5);
figure,imshow(img,[]);
title('Filtered Image');

[rn,cn]=size(img);
imgsiz=rn*cn;
imgv=reshape(img,imgsiz,1);
imgv=double(imgv);

MF=initfcm(ncluster,imgsiz);

% Main loop
for i = 1:max_iter,
    [MF, Cent, Obj(i)] = stepfcm2dmf(imgv,FV,[rn,cn],MF,ncluster,expo,...
        1,1,5);
    
	% check termination condition
	if i > 1,
		if abs(Obj(i) - Obj(i-1)) < 1e-2, break; end,
	end
end
    


figure
subplot(231); imshow(img,[])
title('Test Image');
for i=1:ncluster
    imgfi=reshape(MF(i,:,:),size(img,1),size(img,2));
    subplot(2,3,i+1); imshow(imgfi,[])
    title(['Index No: ' int2str(i)])
end


pause(1)
x = inputdlg('Enter the cluster no. containing the ROI only:');
k = str2double(x);

imgfcm=reshape(MF(k,:,:),size(img,1),size(img,2));
figure,imshow(imgfcm);
title('Segmented Image using Fuzzy C-means');

binaryImage=im2bw(imgfcm);

img = ExtractNLargestBlobs(binaryImage, 2);

img1 = bwareaopen(img, 500);
figure,
imshow(img1);
title('Filtered Segmented Image');

beta=0.5;

img=imgg;
imgfcm=img1;
img=double(img);

se=5;       %template radius for spatial filtering
sigma=2;    %gaussian filter weight
d0=.5;      %fuzzy thresholding
epsilon=1.5;    %Dirac regulator

%adaptive definition of penalizing item mu

u=(d0<=imgfcm);
bwa=bwarea(u);  %area of initial contour
bw2=bwperim(u);
bwp=sum(sum(bw2));  %peripherium of initial contour
mu=bwp/bwa;     %Coefficient of the internal (penalizing) energy term P(\phi);
timestep=0.2/mu; %The product timestep*mu must be less than 0.25 for stability
%end

fs=fspecial('gaussian',se,sigma);
img_smooth=conv2(double(img),double(fs),'same');
[Ix,Iy]=gradient(img_smooth);
f=Ix.^2+Iy.^2;
g=1./(1+f);  % edge indicator function.


% define initial level set function as -c0, c0 
%   at points outside and inside of a region R, respectively.
u=u-0.5;
u=4*epsilon*u;
sls(1,:,:)=double(u);

lambda=1/mu;
nu=-2*(2*beta*imgfcm-(1-beta));
%Note: Choose a positive(negative) alf if the initial contour is
% outside(inside) the object.

% start level set evolution
bGo=1;
nTi=0;
while bGo
    u=EVOLUTION(u, g, lambda, mu, nu, epsilon, timestep, 100);
    nTi=nTi+1;
    sls(nTi+1,:,:)=u;
    
   
    bGo=0;
end

imgls=u;

figure,imshow(imgg,[]);
title('Final Segmented Image using FCM');
hold on
imgt(:,:)=sls(1,:,:);
contour(imgt,[0 0],'g');

hold off






