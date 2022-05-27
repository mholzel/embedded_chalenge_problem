# Problem Statement

## Disparity Images 

At a high level, the disparity image is just an image showing which pixels in one image match to which pixels in another image. In stereo imaging, this is used for inferring depth. However, since the disparity image is produced by merging two images, there are also two ways of generating it, depending on which image you use as the reference image.

For example, suppose that the tip of a pencil is observed in both a left and right image. After rectification, the tip will occur in the same row, but at different columns. For example, suppose that the pencil tip occurs in column 87 in the left image and column 97 in the right image. Then if we use the left image as the reference, the disparity image at column 87 should have a value of 10. If we use the right image as the reference, then the disparity image at column 97 should have a value of 10. For simplicity, we can call these two distinct ways of producing the disparity image the *left and right disparity images*, respectively.

## The Consistency Check

Now unfortunately, matching is an expensive operation, so we might consider using some approximate methods to generate the disparity image, or there might be other features that make the matching break down (such as texture, occlusions, etc.). Hence if reliability is critical, then we may want to consider generating both a left and right disparity image, and cross-referencing the two. For example, in the previous case, we said that the left disparity image should have a value of 10 in column 87 of the pencil-tip row, whereas the right disparity image should have a value of 10 in column 97. If that case were the case, then that specific pixel in both images would be called *consistent*. 

However, what if right disparity image had a value of 90 in column 97? That would suggest that the left pixel at column 87 matched with column 97 of the right image, but the right pixel at column 97 matched with column 7 of the left image. Therefore, depending on tolerances, we would likely call the pixel at column 87 in the left image *inconsistent*. We should invalidate that pixel in the left disparity image to indicate that the depth information from the pixel is not reliable.  But what about the right pixel at column 97? Well since that pixel had a value of 90, this tells us that it was matched with the left pixel at column 7. It is still totally possible that those two pixels are consistent. So this consistency check is a one-way operation.

One other point that we touched upon was the idea of a tolerance. As with all engineering applications, we never expect the result to be perfect, so we must be able to handle some noise and errors. For example, what if the left disparity at column 87 was 10 and the right disparity at column 97 was 11? or 12? or 13?... This is where the idea of tolerance comes in. We want to be able to accommodate some amount of error, but as is typical, this parameter is often left to the user.

## A More Relaxed Check 

One of difficult area for pixel matching is when there are large swaths of the image with the same pattern. For example, image a black and white checkboard-styled tile floor. Such areas might be problematic because the pixel matching has to figure out which black square in one image corresponds to which black square in the other image. 

The traditional consistency check  

left_disp[col] \approx

 

Most of the work involved in this test involved setting up the shell to run the OpenCL program.

At a high-level I see two ways of approaching this problem. The first way is to assume that the
match is perfect in the left (or right) image, and then to
