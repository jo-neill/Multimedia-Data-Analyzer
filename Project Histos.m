l = 180;	% the length of the image 
w = 240;	% the width of the image

RGBim1 = imread('no filter.jpg');
RGBim2 = imread('bleached.jpg');
RGBim3 = imread('antique.jpg');
RGBim4 = imread('dublin.jpg');
RGBim5 = imread('super fade.jpg');
RGBim6 = imread('noir.jpg');

im1 = rgb2hsv(RGBim1);
h1 = im1(:, :, 1);
s1 = im1(:, :, 2);
v1 = im1(:, :, 3);
im2 = rgb2hsv(RGBim2);
h2 = im2(:, :, 1);
s2 = im2(:, :, 2);
v2 = im2(:, :, 3);
im3 = rgb2hsv(RGBim3);
h3 = im3(:, :, 1);
s3 = im3(:, :, 2);
v3 = im3(:, :, 3);
im4 = rgb2hsv(RGBim4);
h4 = im4(:, :, 1);
s4 = im4(:, :, 2);
v4 = im4(:, :, 3);
im5 = rgb2hsv(RGBim5);
h5 = im5(:, :, 1);
s5 = im5(:, :, 2);
v5 = im5(:, :, 3);
im6 = rgb2hsv(RGBim6);
h6 = im6(:, :, 1);
s6 = im6(:, :, 2);
v6 = im6(:, :, 3);

fig1 = figure();
subplot(3,2,1);
histogram(h1);
title('no filter');
subplot(3,2,2);
histogram(h2);
title('bleached');
subplot(3,2,3);
histogram(h3);
title('antique');
subplot(3,2,4);
histogram(h4);
title('dublin');
subplot(3,2,5);
histogram(h5);
title('super fade');
subplot(3,2,6);
histogram(h6);
title('noir');

fig2 = figure();
subplot(3,2,1);
histogram(s1);
title('no filter');
subplot(3,2,2);
histogram(s2);
title('bleached');
subplot(3,2,3);
histogram(s3);
title('antique');
subplot(3,2,4);
histogram(s4);
title('dublin');
subplot(3,2,5);
histogram(s5);
title('super fade');
subplot(3,2,6);
histogram(s6);
title('noir');

fig3 = figure();
subplot(3,2,1);
histogram(v1);
title('no filter');
subplot(3,2,2);
histogram(v2);
title('bleached');
subplot(3,2,3);
histogram(v3);
title('antique');
subplot(3,2,4);
histogram(v4);
title('dublin');
subplot(3,2,5);
histogram(v5);
title('super fade');
subplot(3,2,6);
histogram(v6);
title('noir');


