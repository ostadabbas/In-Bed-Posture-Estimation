function Iout = TransformImg(I, transVec, rot)
%  Iout = TransformImg(I, transVec, rot)
% this function transform rotate the image along the center part then 
% transform the image as the transVec(x,y) indicated. 
% Return the center part only. This is only designed for small variation for
% our purpose only. Well-rounded one need further refinement. 
if nargin >3
    error('myfuns:TransformImg:TooManyInputs',...
        'requires at most 1 optional inputs and 2 required ones');
end

switch nargin
    case 2
        rot = 0;        
end

[m,n,c] = size(I);
width = min(m,n);
if width< max(transVec)
	error( 'Too large translation occurred!');
end

% pad both sides with width , n in our case. 
Ipd = padarray(I,[width,width],'symmetric','both');

if 0~=rot 
	Ir = imrotate(Ipd,rot, 'bicubic', 'crop');
else
	Ir = Ipd;
end

[mr,nr,cr] = size(Ir);
centerRC = round([mr/2,nr/2]); 		% row column 
if mod(m,2)	%odd rows. 
	indFirR = centerRC(1)-(m-1)/2-transVec(2);
	indEndR = centerRC(1)+(m-1)/2-transVec(2);
else			% nod odd, center bias to upper part. 
	indFirR = centerRC(1)+1-m/2-transVec(2);
	indEndR = centerRC(1)+m/2-transVec(2);
end

if mod(n,2)	%odd rows. 
	indFirC = centerRC(2)-(n-1)/2-transVec(1);
	indEndC = centerRC(2)+(n-1)/2-transVec(1);
else			% nod odd, center bias to upper part. 
	indFirC = centerRC(2)+1-n/2-transVec(1);
	indEndC = centerRC(2)+n/2-transVec(1);
end


Iout = Ir(indFirR:indEndR,indFirC:indEndC,:);
