%
% This is a simple function using different
% "strip-mining" lengths in vectorising a
% Monte Carlo computation
%

function test(M2)

%
%
%

M = 10^7;

sum1 = 0;
sum2 = 0;

for m = 1:M2:M
  m2 = min(M+1-m,M2);

  x = rand(1,m2);
  sum1 = sum1 + sum(x);
  sum2 = sum2 + sum(x.^2);
end

ave = sum1/M
std = sqrt(sum2/M - ave^2)

end

