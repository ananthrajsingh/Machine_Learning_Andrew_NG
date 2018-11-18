function vectorImage = imageTo20x20gray(fileName, cropPercentage = 0, rotStep = 0)

%IMAGETO20X20GRAY display reduced image and converts for digit classification
%
% Sample usage: 
%       imageTo20x20Gray('myDigit.jpg', 100, -1);
%
%       First parameter: Image file name
%             Could be bigger than 20 x 20 px, it will
%             be resized to 20 x 20. Better if used with
%             square images but not required.
% 
%       Second parameter: cropPercentage (any number between 0 and 100)
%             0  0% will be cropped (optional, no needed for square images)
%            50  50% of available croping will be cropped
%           100  crop all the way to square image (for rectangular images)
% 
%       Third parameter: rotStep
%            -1  rotate image 90 degrees CCW
%             0  do not rotate (optional)
%             1  rotate image 90 degrees CW
%

% Image will be initially read as RGB image
image3DmatrixRGB = imread(fileName);

% Converting to NTSC image YIQ format, i.e. making image grayscale
image3DmatrixYIQ = rgb2ntsc(image3DmatrixRGB);

% Convert to grays keeping only luminance (Y) and discard chrominance (IQ)
image2DmatrixBW = image3DmatrixYIQ(:,:,1);

%getting to know the size of image
oldSize  = size(image2DmatrixBW);

% Obtain crop size toward centered square (cropDelta)
% ...will be zero for the already minimum dimension
% ...and if the cropPercentage is zero, 
% ...both dimensions are zero
% ...meaning that the original image will go intact to croppedImage

cropDelta = floor((oldSize - min(oldSize)) .* (cropPercentage/100));


% What should be the final size of image?
finalSize = oldSize - cropDelta;

% Compute each dimension origin for croping
% This is from right from where the cropping should start
% if total 20 pixels are to be cropped, 10 will be cropped from either side
cropOrigin = floor(cropDelta / 2) + 1;

% Compute each dimension copying size
copySize = cropOrigin + finalSize - 1;

% Copy just the desired cropped image from the original B&W image
croppedImage = image2DmatrixBW(cropOrigin(1):copySize(1), cropOrigin(2):copySize(2));

% Resolution scale factors: [rows cols]
scale = [20 20] ./ finalSize;

% Compute back the new image size (extra step to keep code general)
newSize = max(floor(scale .* finalSize),1); 

% Compute a re-sampled set of indices:
rowIndex = min(round(((1:newSize(1))-0.5)./scale(1)+0.5), finalSize(1));
colIndex = min(round(((1:newSize(2))-0.5)./scale(2)+0.5), finalSize(2));
% Copy just the indexed values from old image to get new image
newImage = croppedImage(rowIndex,colIndex,:);
% Rotate if needed: -1 is CCW, 0 is no rotate, 1 is CW
newAlignedImage = rot90(newImage, rotStep);
% Invert black and white
invertedImage = - newAlignedImage;

% Find min and max grays values in the image
maxValue = max(invertedImage(:));
minValue = min(invertedImage(:));

%compute the range of shade
delta = maxValue - minValue;


% Normalize grays between 0 and 1
normImage = (invertedImage - minValue)/delta;

% Add contrast. Multiplication factor is contrast control.
contrastedImage = sigmoid((normImage -0.5) * 5);

% Show image as seen by the classifier
imshow(contrastedImage, [-1, 1] );

% Output the matrix as a unrolled vector
vectorImage = reshape(contrastedImage, 1, newSize(1)*newSize(2));
end





