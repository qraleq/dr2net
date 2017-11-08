function [ patches, patchesVectorized, Xp, Yp ] = extractImagePatches( image, blockSize, mode, varargin )
%extractImagePatches Summary of this function goes here
%   Detailed explanation goes here

p=inputParser;

p.addRequired('image', @ismatrix);
p.addRequired('blockSize', @(x) isnumeric(x) && (x>0));
p.addRequired('mode', @isstr);
p.addParameter('Overlap', 0, @(x) isnumeric(x) && (x>=0));
p.addParameter('nPatches', 0, @(x) isnumeric(x) && (x>0));

p.parse(image, blockSize, mode, varargin{:});

[imH, imW] = size(image);

if(p.Results.Overlap > (blockSize - 1))
    error('Invalid overlap parameter!')
end

if(strcmp(p.Results.mode, 'seq'))
    
    % define step for patch extraction
    % default is no overlap(q=blockSize) and maximum is q=1 in that case
    % block centered on every pixel is extracted
    q = blockSize - p.Results.Overlap;
    
    [y, x] = meshgrid(1:q:imW-blockSize/2, 1:q:imH-blockSize/2);
    [dY,dX] = meshgrid(0:blockSize-1,0:blockSize-1);
    
    m = size(x(:),1);
    
    % create indexing grids for block extraction
    Xp = repmat(dX,[1 1 m]) + repmat( reshape(x(:),[1 1 m]), [blockSize blockSize 1]);
    Yp = repmat(dY,[1 1 m]) + repmat( reshape(y(:),[1 1 m]), [blockSize blockSize 1]);
    
    % boundary indices condition
    Xp(Xp>imH) = 2*imH-Xp(Xp>imH);
    Yp(Yp>imW) = 2*imW-Yp(Yp>imW);
    
    patches = image(Xp+(Yp-1)*imH);
    
%     h = fspecial('gauss', blockSize, blockSize/2);
%     patches = patches .* repmat(h, 1, 1, size(patches, 3));
    
    patchesVectorized  = reshape(patches, [blockSize^2, m]);

end


if(strcmp(p.Results.mode, 'rand'))
    
    % number of randomly selected image patches
    q = p.Results.nPatches;
    
    % select q random locations in image (upper left block corners)
    x = floor(rand(1,1,q)*(imH-blockSize))+1;
    y = floor(rand(1,1,q)*(imW-blockSize))+1;
    
    % create rectangular mesh wxw
    [dY,dX] = meshgrid(0:blockSize-1,0:blockSize-1);
    
    % generate matrices containting block locations
    Xp = repmat(dX, [1 1 q]) + repmat(x, [blockSize blockSize 1]);
    Yp = repmat(dY, [1 1 q]) + repmat(y, [blockSize blockSize 1]);
    
    % extract and vectorize blocks
    patches = image(Xp+(Yp-1)*imH);
    
    
    
%     h = fspecial('log', blockSize);
%     patches = patches .* repmat(h, 1, 1, size(patches, 3));
    
    patchesVectorized = reshape(patches, [blockSize^2, q]);
    
%     patchesVectorized = unique(patchesVectorized', 'rows')';
    
end

end

