close all
clearvars
clc

addpath('utilities')

blockSize = 16;
% create random measurement matrix with measurement_rate*blockSize^2
measurement_rate = 0.25;
phi = randn(ceil(measurement_rate*blockSize^2), blockSize^2);
% save('phi', 'phi')

psi = kron(dctmtx(blockSize)',dctmtx(blockSize)')
save('psi', 'psi')


psi_1d = full(wmpdictionary(blockSize, 'lstcpt', {'dct'}));
psi = kron(psi_1d, psi_1d);

%% CREATE TRAINING DATASET
% get all image names from directory
filenames = dir('images\training');
filenames = filenames(~[filenames.isdir])

% set block size and number of images from which nPatches at nScales will
% be extracted
nImages = length(filenames);
nPatches = 80;
nScales = 4;
patches = [];

if(nImages>length(filenames))
    error('Number of used images must be smaller than total number of images!')
end

% extract patches from images
for i = 1:nImages
    image = im2double(imread([filenames(i).folder, '/', filenames(i).name]));
    
    if(size(image,3)>2)
        image = rgb2gray(image);
    end
    
    for scale = 1:nScales
        patches_temp = extractImagePatches(imresize(image, 1./scale), blockSize, 'rand', 'nPatches', nPatches);
        patches = cat(3, patches, patches_temp);
    end
end

% calculate measurements from patches
for it = 1:size(patches, 3)
    patch = reshape(patches(:,:,it), [1, size(patches(:,:,it), 1)*size(patches(:,:,it), 1)]);
    
    patches_vec(:,:,it) = patch(:);
    
    measurement = phi*patch(:);
    measurements(:,:,it)=measurement';
end

% save measurement matrix and train_dataset
save('training_dataset', 'measurements', 'patches_vec')


%% CREATE VALIDATION DATASET
% get all image names from directory
filenames = dir('images\validation');
filenames = filenames(~[filenames.isdir])

% set block size and number of images from which nPatches at nScales will
% be extracted
blockSize = 16;
nImages = length(filenames);
nPatches = 80;
nScales = 4;
patches = [];

if(nImages>length(filenames))
    error('Number of used images must be smaller than total number of images!')
end

% extract patches from images
for i = 1:nImages
    image = im2double(imread([filenames(i).folder, '/', filenames(i).name]));
    
    if(size(image,3)>2)
        image = rgb2gray(image);
    end
    
    for scale = 1:nScales
        patches_temp = extractImagePatches(imresize(image, 1./scale), blockSize, 'rand', 'nPatches', nPatches);
        patches = cat(3, patches, patches_temp);
    end
end

% calculate measurements from patches
for it = 1:size(patches, 3)
    patch = reshape(patches(:,:,it), [1, size(patches(:,:,it), 1)*size(patches(:,:,it), 1)]);
    
    patches_vec(:,:,it) = patch(:);
    
    measurement = phi*patch(:);
    measurements(:,:,it)=measurement';
end

% save measurement matrix and train_dataset
save('validation_dataset', 'measurements', 'patches_vec')

%% EVALUATE LEARNED MODEL
clearvars

load phi_inv
% load phi_inv2
load phi
% phi = phi';

patches_temp = []
patches_test = []

blockSize = 16

filenames = dir('images\testing');
filenames = filenames(~[filenames.isdir])

% load test image i
for i = 30
   
    image = im2double(imread([filenames(i).folder, '/', filenames(i).name]));
    
    %     image = imresize(image, [256, 256]);  

    if(size(image,3)>2)
        image = rgb2gray(image);
    end
    
    [patches_temp, patches_temp_vectorized, Xp, Yp] = extractImagePatches(image, blockSize, 'seq', 'Overlap', 0);
    patches_test = cat(3, patches_test, patches_temp);
end

[imH, imW] = size(image)

% calculate measurement from image patches and calculate reconstruction
% using learned pseudo-inverse matrix
for it = 1:size(patches_test, 3)
    patch_test = reshape(patches_test(:,:,it), [1, size(patches_test(:,:,it), 1)*size(patches_test(:,:,it), 1)]);
    measurement = phi*patch_test(:);
    
    rec(:,:,it) = (measurement'*phi_inv);
end

PA = reshape(rec, [blockSize blockSize size(patches_test, 3)]);
reconstruction = zeros(imH, imW);

for i=1:size(patches_test, 3)
    x = Xp(:,:,i);
    y = Yp(:,:,i);
    
    reconstruction(x+(y-1)*imH) = reconstruction(x+(y-1)*imH) + PA(:,:,i);
end

figure,
subplot(121), imagesc(image), title('Original'), axis image
subplot(122), imagesc(reconstruction), title('Reconstruction'), axis image

psnr = psnr(image, reconstruction)




