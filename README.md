# Drawing with Fingers (Python+OpenCV)
>Note: I'm using Python 3.7 and OpenCV 3.4.3, for people using others versions of these some instructions may be different.

## Enviroment
- OS: Windows 10
- Platform: Python 3.7
- Libraries:
       - OpenCV 3.4.3
       - Numpy
       - OS
       - Math
       - Pynput
     
## How to use the program?
- Run it in Python
- Press `'s'` to set the values of Hue, Saturation and Value
- Press `'c'` to capture the Background
- Press `'a'` to activate the hand drawing
- Press `'ESC'` to exit

## Process
#### Using Camshift algorithm

As we know, the Camshift algorithm consist of taking the first frame of a video to map its colors with an histogram to later on track these all around the video, so to start you need to run the program in Python and inmediatly put your hand in front of the camera.
After this, it must open two windows and some Trackbars. Those ones are for setting the mask for the histogram that we talk before.

![Alt text](Files/)

Move the Ranges of Hue, Saturation and Value to define just your hand in the "Filter" window until you see something like this:

![Alt text](Files/)

When you're done, press `'s'` to set the mask, the windows must close and it will pop-up Paint and a new Trackbar called Threshold.

#### Capture the Background
I use the Background substraction method called **Gaussian Mixture-based Background/Foreground Segmentation Algorithm** wich captures the Background and compares it with the Foreground to subtracts it. Basically it take a frame of the video and compares it with the rest of them.
Here i used the OpenCV's built-in function `BackgroundSubtractorMOG2` to do the job:

```python
bgCap = cv.createBackgroundSubtractorMOG2(0,50)
```
Build a Background subtractor model:

```python
fgMask = bgCap.apply(frame, learningRate = 0)
```
And apply it to the frame:

```python
res = cv.bitwise_and(frame, frame, mask = fgMask)
```

All to get the Foreground image:

![Alt text](Files/)

So to do that, following our "User Guide", you need to first take your hand off the camera vision and next click on the window of the Trackbar and press `'c'`. It must open four new windows that shows us different things of our process like the Background Subtraction one, a Threshold, the Result of the process and the Original one.

#### Setting the image for the final Result
The first thing that I do is create a black image with Numpy:

```python
img = np.zeros(frame.shape, np.uint8)
```

This will help us to see our final result in a nice black Background.
The next stuff is to crop the Background subtaction result with the square that the Camshift algorithm makes:

```python
chanCount = mask.shape[2]
ignoreColor = (255,) * chanCount
cv.fillConvexPoly(img, pts, ignoreColor)
res = cv.bitwise_and(mask, img)
```

It will help us to track just the hand and try to doesn't detect the arm too.
Now I apllied a Filter that consist in different parts:

```python
resMask = cv.dilate(res, ker, iterations = 1)
resMask = cv.morphologyEx(resMask, cv.MORPH_OPEN, ker)
resMask = cv.medianBlur(resMask, 15)
resMask = cv.cvtColor(resMask, cv.COLOR_BGR2GRAY)
_,rThresh = cv.threshold(resMask, vThresh, 255, cv.THRESH_BINARY)
```

It consist in a Dilation of the image that basically helps with noise in the shape of the hand, an Opening Process that helps to reduce the noise in the Backgound of the image, a Median Blur that smooth the edges of the hand to later convert it to Grayscale to finally make a threshold that give us a binary image of the hand. It looks like this:

![Alt text](Files/)

#### Finding and Drawing Contours, Centroid and Farthest Point
Our image is now a binary version of the hand that is cropped for just detect the hand. The next step is to find contours and define the maximum one:

```python
_,con,_ = cv.findContours(rThresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
max_i = 0
max_area = 0
for i in range(len(con)):
    hand = con[i]
    area_hand = cv.contourArea(hand)
    if area_hand > max_area:
    max_area = area_hand
    max_i = i
```

Now we need to actually find the fingers using Convex Hull and Convexity Defects operations, it will return to us the Start, End and Far Points of each finger:

```python
hull = cv.convexHull(max_con, returnPoints = False)
defects = cv.convexityDefects(max_con, hull)
if defects is None:
    defects = [0]
    num_def = 0
else:
    num_def = 0
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(max_con[s][0])
        end = tuple(max_con[e][0])
        far = tuple(max_con[f][0])
```

The next thing is to do simple Math to use the triangles made for the fingers:

```python
a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
s = (a+b+c)/2
```

And apply the cosine rule to determine the angles of the triangles. We must ignore the angles bigger than 90 degrees because of the possible noise to draw the dots for the defects:

```python
angle = math.acos((b**2 + c**2 - a**2) / (2*b*c)) * 57

if angle <= 90 and d > 30:
    num_def += 1
    cv.circle(res, far, 3, (255,0,0), -1)
```

Finally we draw the lines around the hand:

```python
cv.line(res, start, end, (0,0,255), 2)
```

Now for the centoid we use the function Moments and do the logic operations to find the center of mass of our hand in the two coordinates:

```python
moment = cv.moments(max_con)
if moment is None:
    cx = 0
    cy = 0
else:
    cx = 0
    cy = 0
    if moment["m00"] != 0:
        cx = int(moment["m10"] / moment["m00"])
        cy = int(moment["m01"] / moment["m00"])
````

Drawing the centroid:

```python
cv.circle(res, (cx,cy), 5, (0,255,0), 2)
cv.circle(mask, (cx,cy), 5, (0,255,0), 2)
```

The last thing to consider is the farthest point form our centroid. This is because we want to draw with our finger making it the farthest point.
It is done with some claculus:

```python
s = defects[:,0][:,0]

x = np.array(max_con[s][:,0][:,0], dtype = np.float)
y = np.array(max_con[s][:,0][:,1], dtype = np.float)

xp = cv.pow(cv.subtract(x, cx), 2)
yp = cv.pow(cv.subtract(y, cy), 2)

dist = cv.sqrt(cv.add(xp, yp))
dist_max_i = np.argmax(dist)

if dist_max_i < len(s):
    farthest_defect = s[dist_max_i]
    farthest_point = tuple(max_con[farthest_defect][0])
    
cv.line(res, (cx,cy), farthest_point, (0,255,255), 2)
cv.line(mask, (cx,cy), farthest_point, (0,255,255), 2)
```

#### Recognizing the Gestures and controlling the Brush:
This step is very basic. If you're paying attention, in the last part of the code with have a variable called `'num_def'` and it was counting the number of defects that our code detect in the Convex Hull so we're going to take adventage of this and use for Recognizing the Gestures that we do in front of the camera.
Let see how it works:

```python
if num_def == 1:
    cv.putText(frame, "2", (0,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3, cv.LINE_AA)
    if count == 0:
        mouse.release(Button.left)
        mouse.position = (341, 82)
        mouse.press(Button.left)
        mouse.release(Button.left)
        mouse.position = farthest_point
        count = 1
                
     elif num_def == 2:
        cv.putText(frame, "3", (0,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3, cv.LINE_AA)
        if count == 0:
            mouse.release(Button.left)
            mouse.position = (254, 106)
            mouse.press(Button.left)
            mouse.release(Button.left)
            mouse.position = farthest_point
            count = 1
                
     elif num_def == 3:
        cv.putText(frame, "4", (0,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3, cv.LINE_AA)
        if count == 0:
            mouse.release(Button.left)
            mouse.position = (837, 69)
            mouse.press(Button.left)
            mouse.release(Button.left)
            mouse.position = farthest_point
            count = 1
                
     elif num_def == 4:
        cv.putText(frame, "5", (0,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3, cv.LINE_AA)
        if count == 0:
            mouse.release(Button.left)
            mouse.position = (772, 69)
            mouse.press(Button.left)
            mouse.release(Button.left)
            mouse.position = farthest_point
            count = 1
                
     else:
     cv.putText(frame, "1", (0,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3, cv.LINE_AA)
     mouse.position = farthest_point
     mouse.press(Button.left)
     count = 0
```

As you can see it consider the number of defects that are around the Convex Hull and makes different things:
- If you shows to it **one**, the mouse will follows the movement of your finger to draw
- If you shows to it **two**, it will select the **Brush tool** and comes back to your finger automatically
- If you shows to it **three**, it will select the **Eraser tool** and comes back to your finger automatically
- If you shows to it **four**, it will **change to the red color**
- If you shows to it **five**, it will **change to the black color**

>Disclaimer: As the system works with coordinates maybe it won't work propertly but you can only change the coordinates values inside the mouse position brackets

When you are ready to begin the gesture recognition and the contours around your hand are drawn correctly, press `'a'` to activate the tracking. The code will actually give you some feedback to the "Live" window showing you how many fingers are you showing, however, how we are counting defects, as you can imagine is no difference if you show to it one finger o cero fingers more than a bit of noise from the claculus of the farthest point.
The last thing to consider is that if you want to recover the mouse controll just take your hand off the camera vision.

![Alt text](Files/)

To exit the program click on any window except for the Paint one and press `'ESC'`.
----------------------
##References & Tutorials

1. OpenCV documentation: https://docs.opencv.org/3.4.3/
2. Camshift Algorithm: https://docs.opencv.org/3.4.3/db/df8/tutorial_py_meanshift.html
3. Background Subtaction: https://docs.opencv.org/3.4.3/db/d5c/tutorial_py_bg_subtraction.html
4. Cropping image with invisible square: https://stackoverflow.com/questions/15341538/numpy-opencv-2-how-do-i-crop-non-rectangular-region/15343106
5. Image filter (Smoothing): https://docs.opencv.org/3.4.3/d4/d13/tutorial_py_filtering.html
6. Image filter (Reducing noise): https://docs.opencv.org/3.4.3/d9/d61/tutorial_py_morphological_ops.html
7. Contours and Centroid: https://docs.opencv.org/3.4.3/dd/d49/tutorial_py_contour_features.html
8. Finding Farthest Point: https://dev.to/amarlearning/finger-detection-and-tracking-using-opencv-and-python-586m
9. Gestures: https://github.com/Sadaival/Hand-Gestures
10. Image Thresholding: https://docs.opencv.org/3.4.3/d7/d4d/tutorial_py_thresholding.html
11. Color filtering: https://docs.opencv.org/3.4.3/df/d9d/tutorial_py_colorspaces.html
12. Python learning in general: https://www.youtube.com/user/sentdex
