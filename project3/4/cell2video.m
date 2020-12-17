function vid = cell2video(cellar)
  % get number of frames(cells) in cell array
  [frames, ~] = size(cellar);
  for i = 1:frames
    % write individual frames into video structure
    vid(i).cdata = uint8(cellar{i,1});
    vid(i).colormap = [];
  end
end