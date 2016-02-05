%% Cleanup
clear all;close all;clc;

%% Load and view data
addpath(genpath(cd));

imageRaw = im2double(imread('cameraman.tif'));
figure(1);imagesc(imageRaw,[0,1]);colormap(gray);axis image;title('Raw Image');

imageRawCut = imageRaw;%(1:end,50:200);
figure(2);imagesc(imageRawCut,[0,1]);colormap(gray);axis image;title('Raw Image, Zoom');

imageRawCutNoise = imnoise(imageRawCut,'gaussian',0,0.005);
figure(3);imagesc(imageRawCutNoise,[0,1]);colormap(gray);axis image;title('Noisy Image, Zoom');

%% Algorithms
tol = 1e-5; %specify how accurate the solution should be
dimsU = size(imageRawCutNoise); %specify size of the expected solution

parameterL2L2 = 1; %regularization parameter for L2L2 model
parameterL2TV = 0.15; %regularization parameter for L2TV model
parameterL1TV = 0.75; %regularization parameter for L1TV model
tic;
resultL2L2 = DenoisingFirstOrder(imageRawCutNoise,dimsU,tol,parameterL2L2,'regularizer','L2','dataterm','L2');
toc;
%%

resultL2TV = DenoisingFirstOrder(imageRawCutNoise,dimsU,tol,parameterL2TV,'regularizer','TV','dataterm','L2');
resultL1TV = DenoisingFirstOrder(imageRawCutNoise,dimsU,tol,parameterL1TV,'regularizer','TV','dataterm','L1');

%% View results
figure(4);imagesc(resultL2L2,[0,1]);colormap(gray);axis image;title('Result L2L2 model');
figure(5);imagesc(resultL2TV,[0,1]);colormap(gray);axis image;title('Result L2TV model');
figure(6);imagesc(resultL1TV,[0,1]);colormap(gray);axis image;title('Result L1TV model');

%% Inpainting
tol = 1e-5; %specify how accurate the solution should be
dimsU = size(imageRawCutNoise); %specify size of the expected solution

characteristicFunction = rand(dimsU);
characteristicFunction = characteristicFunction<0.1;

nPx = prod(dimsU);

figure(7);imagesc(characteristicFunction,[0,1]);colormap(gray);axis image;title('Characteristic Function');

% create downsampling operator
downsamplingOperator = spdiags(characteristicFunction(:),0,numel(characteristicFunction),numel(characteristicFunction));
downsamplingOperator(characteristicFunction(:)==0,:) = [];

% create blurring operator
weights = [1/20 4/20 10/20 4/20 1/20;1/20 4/20 10/20 4/20 1/20];
[ blurrOperator] = generateBlurrND(dimsU,weights );

inputF = downsamplingOperator*imageRawCutNoise(:);

figure(8);imagesc(reshape(downsamplingOperator'*inputF,dimsU),[0,1]);colormap(gray);axis image;title('Input Data');

%% Algorithms
parameterL2L2 = 1;
parameterL2TV = 0.05;

resultL2L2inpainting = DenoisingFirstOrder(inputF,dimsU,tol,parameterL2L2,'regularizer','L2','dataterm','L2','operatorK',downsamplingOperator*blurrOperator);
resultL2TVinpainting = DenoisingFirstOrder(inputF,dimsU,1e-7,parameterL1TV,'regularizer','TV','dataterm','L1','operatorK',downsamplingOperator*blurrOperator);

%% View results
figure(9);imagesc(resultL2L2inpainting,[0,1]);colormap(gray);axis image;title('Result L2L2 model');
figure(10);imagesc(resultL2TVinpainting,[0,1]);colormap(gray);axis image;title('Result L1TV model');