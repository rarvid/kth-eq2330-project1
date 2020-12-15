% extract first 50 frames from foreman video
% cell array with 50 frames(cells) - 176x144 double array
foreman = yuv_import_y('foreman_qcif.yuv',[176 144],50);

% cell array with 50 frames(cells), each frame contains 18x22 cell array,
% each cell array cotains 8x8 double arrays - DCT block. DCT2 has been
% applied to each DCT Block individually
dct8 = cellfun(@dct_blocks,foreman,'UniformOutput',0);

% cell array with 4 videos quantized with differnet quantization levels 
% each with 50 frames where each frames has a 18x22 cell array with 8x8
% double arrays - Quantized DCT block
var_quant = quant_mult(dct8);

% cell array with 4 videos each with 50 frames, each frames is a 176x144
% double array that has been obtained by reconstructing the 8x8 DCT Blocks
% using IDCT2
recon = reconstruct(var_quant);

% cell array with 4 videos, each with 50 frames, where each frame is a
% 176x144 uint8 array
ui8rec = double2uint8(recon);

% A matlab video format structure to view frames as videos
vid = cell2video(ui8rec{4,1});

% Takes a cell with 'i' frames and converts it into a matlab video structure
function vid = cell2video(cellar)
  % get number of frames(cells) in cell array
  [frames, ~] = size(cellar);
  for i = 1:frames
    % write individual frames into video structure
    vid(i).cdata = cellar{i,1};
    vid(i).colormap = [];
  end
end


% Takes a cell array with 'i' videos each with 'j' frames and converts the
% frames from double to uint8
function rund = double2uint8(vals)
  [levels, ~] = size(vals);
  for i = 1:levels
    [frames, ~] = size(vals{i,1});
    for j = 1:frames
      rund{i,1}{j,1} = uint8(vals{i,1}{j,1});
    end
  end
end

% takes a cell array with 'i' videos each with 'j' frames that have been
% divided into 8x8 DCT blocks and reconstruct each DCT Block with IDCT2 and
% combines the reconstructed blocks of each frame
function rec = reconstruct(coefs)
  [levels, ~] = size(coefs);
  for i = 1:levels
    [frames, ~] = size(coefs{i,1});
    for j = 1:frames
      % the j-th frame of the i-th video
      rec{i,1}{j,1} = cellfun(@idct2, coefs{i,1}{j,1}, 'UniformOutput', 0);
      rec{i,1}{j,1} = cell2mat(rec{i,1}{j,1});
    end
  end

end

% applies quantization with step size 2^3..6 on the same given video and
% returns a cell with videos that have the quantization applied to their
% DCT 8x8 blocks in each frame
function mult = quant_mult(vid)
  % apply quantization to all frames using varying stepsizes
  for i = 3:6
    step_size = 2.^i;
    mult{i-2,1} = quant_frames(vid, step_size);
  end
end

% applies quantizer to video with 'i' frames that have been split into
% 8x8 DCT blocks  
function q = quant_frames(vid,step)
  % get number of frames
  [frames, ~] = size(vid);
  % apply to all frames
  for i = 1:frames
    % number of blocks in frame in x and y coord
    [sizex , sizey] = size(vid{i,1}); 
    % create step size for each block (this is a cellfun thing)
    step_size = num2cell(step * ones(sizex, sizey)); 
    % apply quantizer to each 8x8 block in each frame to all frames
    q{i,1} = cellfun(@quantizer, vid{i,1}, step_size, 'UniformOutput', 0); 
  end
  
end

% takes  cell array that contains frames, splits each frame into 8x8 Blocks
% and applies DCT2 to each block
function res = dct_blocks(im)
  % get dimensions of frame
  [sizex, sizey] = size(im);
  % calculate number of 16x16 blocks needed
  xvector_8 = 8*ones(1, sizex ./ 8 );
  yvector_8 = 8*ones(1, sizey ./ 8 );
  % split frame into 16x16 blocks
  blocks_8 = mat2cell(im,xvector_8,yvector_8);
  res = cellfun(@dct2, blocks_8, 'UniformOutput', 0);
end
