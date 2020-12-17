% Uniform quantzier funtction


function y = quantizer(x, step_size)
  y = step_size * floor(x / step_size + 0.5);
end