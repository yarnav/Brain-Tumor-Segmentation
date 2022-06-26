% All inputs are passed to final_OpeningFcn via varargin.
function varargout = final(varargin)


% Initialization code
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @final_OpeningFcn, ...
                   'gui_OutputFcn',  @final_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end


% --- Attaching all scripts to Figure. Executes just before final is made visible.
function final_OpeningFcn(hObject, eventdata, handles, varargin)
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to final (see VARARGIN)

% Choose default command line output for final
handles.output = hObject;
ss=ones(300,300);
axes(handles.axes1);
imshow(ss);
axes(handles.axes2);
imshow(ss);
axes(handles.axes3);
imshow(ss);
axes(handles.axes4);
imshow(ss);
axes(handles.axes5);
imshow(ss);
axes(handles.axes6);
imshow(ss);
axes(handles.axes7);
imshow(ss);
axes(handles.axes8);
imshow(ss);


% Update handles structure
guidata(hObject, handles);


% --- Outputs from this function are returned to the command line.
function varargout = final_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in radiobutton1.
function radiobutton1_Callback(hObject, eventdata, handles)
%% Choosing the image
[filename,pathname] = uigetfile(...
    '*.jpg;*.tif;*.png;*.jpeg;*.bmp;*.pgm;*.gif','pick an imgae');
file = fullfile(pathname,filename);

   img = imread(file);
   axes(handles.axes1);
   imshow(img);
   title('Test Image');
   
   handles.img=img;

% Update handles structure
guidata(hObject, handles);




function radiobutton2_Callback(hObject, eventdata, handles)
img=handles.img;
%% Preprocessing - Image Enhancement
if size(img,3) == 3
    
    img=rgb2gray(img);
    
end

axes(handles.axes2);
imshow(img);
title('Gray Image');

EI=imadjust(img);
axes(handles.axes2);
imshow(EI);
title('Enhanced Image');

handles.EI=EI;

% Update handles structure
guidata(hObject, handles);




function radiobutton3_Callback(hObject, eventdata, handles)
EI=handles.EI;

%% Feature Extraction using GLCM
g = graycomatrix(EI);
stats = graycoprops(g,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
set(handles.edit1,'string',Contrast);
Correlation = stats.Correlation;
set(handles.edit2,'string',Correlation);
Energy = stats.Energy;
set(handles.edit3,'string',Energy);
Homogeneity = stats.Homogeneity;
set(handles.edit4,'string',Homogeneity);

FV=[Contrast Correlation Energy Homogeneity];
handles.FV=FV;

% Update handles structure
guidata(hObject, handles)




function radiobutton4_Callback(hObject, eventdata, handles)
EI=handles.EI;

%% Generate Template based on reduced gray level intensity
TI = uint8(floor(double(EI)/8));
axes(handles.axes3);
imshow(TI);
title('Image with Gray Level 32');


%% Coarse Image 
im=bitxor(EI,TI);
axes(handles.axes4);
imshow(uint8(im));
title('Coarse Image');

handles.im=im;

% Update handles structure
guidata(hObject, handles)




function radiobutton5_Callback(hObject, eventdata, handles)
im=handles.im;
FV=handles.FV;

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
    %================ for gray image
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
        if (vec_mean==old_mean | itr>105)
            break;
        end
    end
    label_im = reshape(label_vector,size(gray));
    axes(handles.axes5);
    imshow(label_im,[]);
    title('Segmented Image using K-means');
    
 %% Segmentation using Fuzzy C-means based on GLCM Features
 
  ncluster=5;
  
  imgg=label_im,[];
    
  expo=2;

  max_iter=100;
  
img=wiener2(imgg,5);

[rn,cn]=size(img);
imgsiz=rn*cn;
imgv=reshape(img,imgsiz,1);
imgv=double(imgv);

MF=initfcm(ncluster,imgsiz);

% Main loop
for i = 1:max_iter,
    [MF, Cent, Obj(i)] = stepfcm2dmf(imgv,FV,[rn,cn],MF,ncluster,expo,1,1,5);
    
	% termination condition
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
x = inputdlg('Enter the cluster no. containing the ROI');
k = str2double(x);

imgfcm=reshape(MF(k,:,:),size(img,1),size(img,2));
axes(handles.axes6);
imshow(imgfcm);
title('Segmented Image using Fuzzy C-means');

binaryImage=im2bw(imgfcm);

img = ExtractNLargestBlobs(binaryImage, 2);

img1 = bwareaopen(img, 500);
axes(handles.axes7);
imshow(img1);
title('Filter Segmented Image');

img=imgg;
imgfcm=img1;
img=double(img);

se=5;           %template radius for spatial filtering
sigma=2;        %gaussian filter weight
d0=.5;          %fuzzy thresholding
epsilon=1.5;    %Dirac regulator

%adaptive definition of penalizing item mu
u=(d0<=imgfcm);
bwa=bwarea(u);      %area of initial contour
bw2=bwperim(u);     
bwp=sum(sum(bw2));  %peripherium of initial contour
mu=bwp/bwa;         %Coefficient of the internal (penalizing) energy term P(\phi);
timestep=0.2/mu;    %The product timestep*mu must be less than 0.25 for stability


fs=fspecial('gaussian',se,sigma);
img_smooth=conv2(double(img),double(fs),'same');
[Ix,Iy]=gradient(img_smooth);
f=Ix.^2+Iy.^2;
g=1./(1+f);  % edge indicator function.


% define initial level set function as -c0, c0 
%   at points outside and inside of a region R, respectively.
u=u-0.5;
beta=0.5;
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

axes(handles.axes8);
imshow(imgg,[]);
title('Final Segmented Image');
hold on
imgt(:,:)=sls(1,:,:);
contour(imgt,[1 1],'r');

hold off


% Update handles structure
guidata(hObject, handles)



function edit1_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit2_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit3_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit4_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function edit4_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
