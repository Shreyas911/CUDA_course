
a = importdata('matlab_test_log.csv');

cols = a.colheaders;
data = a.data;

nscore = 0;

for n=1:length(cols)
  if (cols{n}(1:6)==' Score')
    nscore = nscore+1;
  end
end

nvar = length(cols) - nscore - 1;

Y = data(:,end);
bar(Y)
xlabel('test')
ylabel('time')
