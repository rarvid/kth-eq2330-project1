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
rounded_rec = doubleround(recon);

% A matlab video format structure to view frames as videos
vid = cell2video(rounded_rec{4,1});

% Cell array with 4 videos each with 50 PSNR values, i.e. each frame of the
% reconstructions compared with the original 50 frames
psnr_frames = psnr_mult(rounded_rec, foreman);

% cell array with average PSNR of 50 frames from 4 different quantization
% levels
avg_psnr = avg(psnr_frames);

% obtain 16x16 blocks with 4 8x8 DCT blocks 
quanted_16_16 = merge_cells(var_quant);

% entropies of each 16x16 block
entropy_255 = entropy_blocks(quanted_16_16);

% average entropy of each frame
average_entropy = avg_entropy(entropy_255);

% bitrate calculation
kbitrate_s = bitrate(average_entropy);

%  PSNR kBit/s plot
figure('visible','off');
plot(cell2mat(kbitrate_s)', cell2mat(avg_psnr)','LineWidth',1);
title('foreman')
xlabel('kbit/s')
ylabel('PSNR')
p_b_plot = gca;
exportgraphics(p_b_plot, 'PSNR_kBit_s_foreman_plot.png');

function b = bitrate(entr)
  [levels, ~] = size(entr);
  for i = 1:levels
    [frames, ~] = size(entr{i,1});
    duration = frames ./ 30;
    entropy_matrix = cell2mat(entr{i,1});
    cum_entropy = sum(entropy_matrix);
    b{i,1} = cum_entropy ./ duration;
    b{i,1} = b{i,1} ./ 1000; 
  end
end


% Calculate the average entropy of each frame
function avge = avg_entropy(cellar)
  [levels, ~] = size(cellar);
  for i = 1:levels
    [frames, ~] = size(cellar{i,1});
    for j = 1:frames
      matrix = cell2mat(cellar{i,1}{j,1});
      avge{i,1}{j,1} = mean(matrix(:)); 
    end
  end
end

% get entropies of each 16x16 block
function e = entropy_blocks(cellar)
  [levels, ~] = size(cellar);
  for i = 1:levels
    [frames, ~] = size(cellar{i,1});
    for j = 1:frames
      [blocks_x, blocks_y] = size(cellar{i,1}{j,1});
      for k = 1:blocks_x
        for l = 1:blocks_y
          e{i,1}{j,1}{k,l} = entropy_single(cellar{i,1}{j,1}{k,l});
        end
      end
    end
  end
end

function en = entropy_single(array)
  U = unique(array);
  H = histcounts(array, U);
  pdf = H / 256;
  total_entropy = 0;
  for i = 1:length(pdf)
    total_entropy = total_entropy + (pdf(i) .* -log2(pdf(i)));
  end
  en = total_entropy;
end

% merge 4 8x8 cell blocks to obatin 16x16 blocks( needed for entropy calc)
function m = merge_cells(cellar)
  [levels, ~] = size(cellar);
  for i = 1:levels
    [frames, ~] = size(cellar{i,1});
    for j = 1:frames
      % merge 8x8 DCT block together
      cellar{i,1}{j,1} = cell2mat(cellar{i,1}{j,1});
      [resx ,resy] = size(cellar{i,1}{j,1});
      xvector_16 = 16 * ones(1, resx ./ 16);
      yvector_16 = 16 * ones(1, resy ./ 16);
      m{i,1}{j,1} = mat2cell(cellar{i,1}{j,1}, xvector_16, yvector_16);
    end
  end
end


% Calculate average of cells
function a = avg(cellar)
  [levels, ~] = size(cellar);
  for i = 1:levels
    a{i,1} = mean(cell2mat(cellar{i,1}));
  end
end

% returns a cell arrary with 'i' cells with 'j' PSNR values, comparing 'i'
% differnet quantization levels and the original image
function p = psnr_mult(vid, orig)
  [levels, ~] = size(vid);
  for i = 1:levels
    [frames, ~] = size(vid{i,1});
    for j = 1:frames
      psnr_frames{i,1}{j,1} = psnr(vid{i,1}{j,1}, orig{j,1}, 255);
      p = psnr_frames;
    end
  end
end

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
function rund = doubleround(vals)
  [levels, ~] = size(vals);
  for i = 1:levels
    [frames, ~] = size(vals{i,1});
    for j = 1:frames
      rund{i,1}{j,1} = uint8(vals{i,1}{j,1});
      rund{i,1}{j,1} = double(rund{i,1}{j,1});
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
