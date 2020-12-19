function [motions_vectors, residuals, distortion]= SSD(vid)
  [frames,~] = size(vid);
  for i = 1:frames-1
    [pixel_x, pixel_y] = size(vid{i,1});
    current = vid{i+1,1};
    prev = vid{i,1};
    % itterate over all 16x16 blocks of image
    iter_k = 0;
    for k = 1:16:pixel_x
      iter_l = 0;
      for l = 1:16:pixel_y
        % initialize the minimal SSD values and corresponding motion vector
        % coordinate
        min_val = intmax;
        min_x = 0;
        min_y = 0;
        % itterate over displacement range [-10,...,10] x [-10,...,10]
        for dx = -10:10
          for dy = -10:10
            % check for edge cases
            if (k + dx > 0) && (k + 15 + dx <= pixel_x) && (l + dy > 0) && (l + 15 + dy <= pixel_y)
              sqdif = (current(k:k+15,l:l+15) - prev(k+dx:k+15+dx,l+dy:l+15+dy)).^2;
              ssd = mean(sqdif(:));
              if ssd < min_val
                min_val = ssd;
                % min motion vector
                min_x = dx;
                min_y = dy;
                % min error image
                e = current(k:k+15,l:l+15) - prev(k+dx:k+15+dx,l+dy:l+15+dy);
              end
            end
          end
        end
        motions_vectors{i,1}{k - iter_k * 15, l - iter_l * 15} = [min_x, min_y];
        residuals{i,1}{k - iter_k * 15, l - iter_l * 15} = e;
        distortion{i,1}{k - iter_k * 15, l - iter_l * 15} = min_val;
        iter_l = iter_l + 1;
      end
      iter_k = iter_k + 1;
    end
  end
end

function m = merge_16(cellar)
  [frames, ~] = size(cellar);
  for i = 1:frames
    m{i,1} = uint8(cell2mat(cellar{i,1}));
  end
end



% split each frame of video into 16x16 blocks
% use cellfun to apply to multiple levels
function s = split_16(vid)
  [frames, ~] = size(vid);
  for i = 1:frames
    [resx,resy] = size(vid{i,1});
    xvector_16 = 16 * ones(1, resx ./ 16);
    yvector_16 = 16 * ones(1, resy ./ 16);
    s{i,1} = mat2cell(vid{i,1}, xvector_16, yvector_16);
  end 
end