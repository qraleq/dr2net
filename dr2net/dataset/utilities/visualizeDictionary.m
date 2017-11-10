function [] = visualizeDictionary( D )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

blockSize = sqrt(size(D, 1));

nVis = floor(size(D, 2)/blockSize)*blockSize;
dictVisual = col2im(D(:,1:nVis), [blockSize, blockSize], size(D(:,1:nVis)), 'distinct');

% nVis = 64;
% dictVisual = col2im(D(:,1:nVis), [blockSize, blockSize], [128 128], 'distinct');


imagesc(dictVisual), axis image

xticks(0.5:blockSize:size(dictVisual,2))
yticks(0.5:blockSize:size(dictVisual,1))

set(gca,'xticklabel',[])
set(gca,'yticklabel',[])

grid on

ax = gca;
ax.GridColor = 'black';
ax.GridAlpha = 1;
set(gca,'LineWidth', 2);

end

