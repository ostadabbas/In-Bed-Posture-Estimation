function Xpca=ToPCAspace(X,mean,coeff)
% transfer the input matrix to PCA space, 
% input,mean, featrues space center vector
% coeff, the mapping coeff or the loading. Each column is the new principle
% components in original features space
[m n] =size(X);    % 
meanMat = repmat(mean,m,1);
Xpca= (X-meanMat)*coeff;

