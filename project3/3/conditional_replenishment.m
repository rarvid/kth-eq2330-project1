%-----------------%
% Header comments %
%-----------------%

% extract first 50 frames from foreman video
% cell array with 50 frames(cells) - 176x144 double array
foreman = yuv_import_y('mother-daughter_qcif.yuv',[176 144],50);

data_8_8 = cell(4, 1);
for i = 3:6
    % load the DCTed blocks in the data
    % several steps are short cut in this line
    vid = cellfun(@dct_blocks, foreman,'UniformOutput',0);
    % define our step size
    step_size = 2.^i;
    % with our step size, we can quantize the DCTed blocks
    vid = quant_frames(vid, step_size);
    % store the vidéo in the data
    data_8_8{i-2, 1} = vid;
end

% Reconstruct the vidéos according to part 2
% so that we can compute the distortion
recon = reconstruct(data_8_8);
rounded_rec = doubleround(recon);
% rounded_rec is the reconstructed image for the distortion of the
% intra mode

% from 8×8 to 16×16
data = merge_cells(data_8_8);

% Calculate the bit rate with the intra-mode only data
% so that we have the R1 used in the langragian function.
intra_bit_rate = cellfun(@bit_rate, data, 'UniformOutput',0);

% add the encoding of the mode. data will become a data encoded entirely
% in intra mode (extra byte = 1)
for i = 3:6
    for j = 1:size(data{i-2 ,1})
        [Lx, Ly] = size(data{i-2 ,1}{j, 1});
        for u = 1:Lx
            for v = 1:Ly
                
                block = data{i-2,1}{j,1}{u,v};
                data{i-2,1}{j,1}{u,v} = cell(2, 1);
                % assign the code 1 = intra mode
                % as well as the block itself
                data{i-2,1}{j,1}{u,v}{1,1} = 1;
                data{i-2,1}{j,1}{u,v}{2,1} = block;
                
            end
        end
    end
end

% encode the data with intra or copy mode, block-wise
for i = 3:6
    index = i - 2;
    step_size = 2. ^ i;
    
    data{index,1} = encode_video(data{index,1}, foreman, step_size, intra_bit_rate{index,1});
end

% compute the bit rates now with the two-modes brand-new encoding
rates = cellfun(@bit_rate_two_modes, data, 'UniformOutput',0);

% convert the rates (bit/pixel) into raw kbit/s (not per pixel any more)
[w, h] = size(foreman{1,1});
for i = 1:length(rates)
    rates{i,1} = rates{i,1}(1) * ( w * h * length(foreman) * 30 / 1000);
end

% rebuild the original vidéos
recons = cellfun(@reconstruct_two_modes, data, 'UniformOutput',0);

psnrs = psnr_mult(recons, foreman);
%  PSNR kBit/s plot
figure('visible','on');
plot(cell2mat(rates)', cell2mat(psnrs)','LineWidth',1);
title('')
xlabel('kbit/s')
ylabel('PSNR')
p_b_plot = gca;
exportgraphics(p_b_plot, '3_mother_plot.png');


%%%%%%%%%%%%%%%% LOCAL FUNCTIONS %%%%%%%%%%%%%%%%%

% reconstrictopn
function recons = reconstruct_two_modes(vid)
    recons = cell(size(vid));
    
    deux = size(vid{1,1}) .* size(vid{1,1}{1,1}{2,1});
    Lx = deux(1);
    Ly = deux(2);
    [Bx, By] = size (vid{1,1});
    
    for f = 1:length(vid)
        recons{f,1} = zeros(Lx, Ly);
        for i = 1:Bx
            for j = 1:By
                recon_i = 16 * (i-1) + 1;
                recon_j = 16 * (j-1) + 1;
                recons{f,1}(recon_i:recon_i+15, recon_j:recon_j+15) = idct16 (vid{f,1}{i,j}{2,1});
            end
        end
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
  p = avg(p);
end
% Calculate average of cells
function a = avg(cellar)
  [levels, ~] = size(cellar);
  for i = 1:levels
    a{i,1} = mean(cell2mat(cellar{i,1}));
  end
end

% Une fonction des plus affreuses.
function rate = bit_rate_two_modes(vid)
    [len, ~] = size(vid);
    [Lx, Ly] = size(vid{1,1});
    
    total_pixel_count = len * Lx * Ly;
    % This array will store the pixels' values before we compute the
    % entropy on it.
    pixels = zeros(total_pixel_count, 1);
    % We want a 1-dim list
    pixels = pixels (:);
    
    total_entropy = 0.0;
    
    for u = 1:16
        for v = 1:16
            % Compute the entropy of all the (u,v)-th coefficients, amoung
            % all blocks from all frames.
            
            % index in the pixels list.
            idx = 1;
            
            for k = 1:len
                for i = 1:Lx
                    for j = 1:Ly
                        
                        if vid{k, 1}{i,j}{1,1} == 1 % check intra mode
                            pixel = vid{k,1}{i,j}{2,1}(u,v);
                            pixels(idx, 1) = pixel;
                            idx = idx + 1;
                        end
                        
                    end
                end
            end
            
            % Now we can compute the (u,v)-th entropy.
            list = pixels( 1:(idx-1) );
            total_entropy = total_entropy + entropy(list) * (idx-1) / total_pixel_count;
            % It is tricky now because the VLC's average length is no longer
            % the average per-pixel length, as some pixels are shadowed by
            % the copy mode. A corrective factor (idx-1) / total_pixel_count
            % has to be included. All pixels encoded by the VLC take
            % entropy bits and all pixels encoded by the copy mode take
            % zero bit.
        end
        
    end
    
    % Oh yeah! Every block of 16×16 pixels has an extra bit to encode the
    % mode, so each pixel requires 1/16² extra bit.
    rate = total_entropy / 256.0  + 1.0 / 256.0;
    
end
% Adapation of the above function to work with only the intra-mode.
% this function is the first one to be called.
function rate = bit_rate(vid)
    [len, ~] = size(vid);
    [Lx, Ly] = size(vid{1,1});
    
    total_pixel_count = len * Lx * Ly;
    % This array will store the pixels' values before we compute the
    % entropy on it.
    pixels = ones(total_pixel_count, 1);
    % We want a 1-dim list
    pixels = pixels (:);
    
    total_entropy = 0.0;
    
    for u = 1:16
        for v = 1:16
            % Compute the entropy of all the (u,v)-th coefficients, amoung
            % all blocks from all frames.
            
            % index in the pixels list.
            idx = 1;
            
            for k = 1:len
                for i = 1:Lx
                    for j = 1:Ly
                        
                        pixel = vid{k,1}{i,j}(u,v);
                        pixels(idx) = pixel;
                        idx = idx + 1;
                        
                    end
                end
            end
            
            % Now we can compute the (u,v)-th entropy.
            list = pixels( 1:(idx-1) );
            total_entropy = total_entropy + entropy(list);
        end
    end
    
    rate = total_entropy / 256.0;
    
end

function H = entropy(list)
    uni = unique (list);
    occurrences = zeros(size(uni));
    for i = 1:length(list)
        v = list(i);
        index = -1;
        for k = 1:length(uni)
            if uni(k) == v
                index = k;
                break;
            end
        end
        occurrences(index) = occurrences(index) + 1;
    end
    pdf = occurrences ./ length(list);
    H = 0.0;
    for i = 1:length(pdf)
        H = H - pdf(i) * log2(pdf(i));
    end
    
end

function new_vid = encode_video(vid, orig_vid, step_size, intra_bit_rate)
    [len, ~] = size(vid);
    new_vid = cell(len, 1);
    new_vid{1,1} = vid{1,1};
    
    for i = 2:len
        new_vid{i,1} = encode_frame(new_vid{i-1, 1},vid{i,1},orig_vid{i,1},0.2*step_size*step_size,intra_bit_rate);
    end
end

% Une fonction des plus ténèbreuses.
function new_frame = encode_frame(previous, current, curr_original, lambda, R1)
    % previous and current are two consecutive frames (in DCTed
    % form, encoded in mode intra). Cells filled with 16×16 blocks
    
    
    [Lx, Ly] = size(current);
    new_frame = cell(Lx, Ly);
    
    for i = 1:Lx
        for j = 1:Ly
            curr_block = current{i,j}{2,1};
            orig_i = 16 * (i-1) + 1;
            orig_j = 16 * (j-1) + 1;
            orig_block = curr_original(orig_i:orig_i+15, orig_j:orig_j+15);
            
            % distortion of mode intra
            D1 = immse( idct16(curr_block), orig_block);
            % the bit rate of mode intra, R1, is already given in argument
            
            % distortion of mode copy
            prev_block = previous{i,j}{2,1};
            D2 = immse( idct16(prev_block), orig_block);
            % bit rate of mode copy
            R2 = 1 ./ 256; % yep! only one bit of information of 256 pixels!
            
            % the aiguillage is here
            cellule = cell(2,1);
            if D1 + lambda * R1 < D2 + lambda * R2
                % intra mode wins
                % we do nothing as the current frame is already in intra
                % mode. The only thing to do it to copy.
                cellule{1,1} = 1;
                cellule{2,1} = curr_block;
            else
                % copy mode wins
                cellule{1,1} = 2;
                cellule{2,1} = prev_block; % normally that cell should be empty
            end
            new_frame{i,j} = cellule;
        end
    end
    
end

function blk = idct16(array)
    % apply the idct to the four 8×8 sub-blocks
    blk(1:8, 1:8)   = idct2(array(1:8, 1:8));
    blk(9:16, 1:8)  = idct2(array(9:16, 1:8));
    blk(1:8, 9:16)  = idct2(array(1:8, 9:16));
    blk(9:16, 9:16) = idct2(array(9:16, 9:16));
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
  % split frame into 8x8 blocks
  blocks_8 = mat2cell(im,xvector_8,yvector_8);
  res = cellfun(@dct2, blocks_8, 'UniformOutput', 0);
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