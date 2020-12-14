% extract first 50 frames from foreman video
foreman = yuv_import_y('foreman_qcif.yuv',[176 144],50);

% x = imread("airfield512x512.png");
% ext = dct_blocks(x);

ext = cellfun(@dct_blocks,foreman,'UniformOutput',0);

function res = dct_blocks(im) % divide frame into 16x16 blocks
  % get dimensions of frame
  [sizex, sizey] = size(im);
  % calculate number of 16x16 blocks needed
  xvector_8 = 8*ones(1, sizex ./ 8 );
  yvector_8 = 8*ones(1, sizey ./ 8 );
  % split frame into 16x16 blocks
  blocks_8 = mat2cell(im,xvector_8,yvector_8);
  res = cellfun(@dct2, blocks_8, 'UniformOutput', 0);
end
