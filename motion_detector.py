import cv2
first_frame =None
video = cv2.VideoCapture(0) #0 is the index of camera ...for suppose if we have two cameras then 0 will connect to one and 1 will connect to another
 #we will form frames and will run all the frames to have our Video...this would be done either by iteration or by reccursion
while True:
     check , frame = video.read() #check function will correspond to activeness of camera and video.read() would capture the image and store the data in a numpy array
     #print(check)
    # print(frame) #just printing hte array of image(numpyarray)
     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #we are simply converting our image in a black & white model so that face detection could become easier
     gray = cv2.GaussianBlur(gray,(21,21),0) #this would help our program to detect any object easily...for more info google it...
     if first_frame is None:
         first_frame = gray
         continue # consider the background.... the first loop will collect background details and store in first_frame and so untill unless we get a backgruond this loop print no frame
     delta_frame = cv2.absdiff(first_frame,gray)# this is anotherimage created by difference in background ..... for suppose some object enters the space so numpy array would change and hence a diffrence array would be created giving us a new anotherimage
     thresh_frame = cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]#this is a threshold set up...If the diffrence betn the pixels of any certain region is more than 30 show it white (255 is code of white if we"ll give 252 green will appoear)..and [1] is beacause threshold gives two values amd here we need the second...
     thresh_frame = cv2.dilate(thresh_frame,None, iterations = 2)
     (_,cnts,_) = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
     for contours in cnts :
         if cv2.contourArea(contours)<1000:
             continue
         (x,y,w,h)=cv2.boundingRect(contours)
         cv2.rectangle(frame,(x,w),(x+w,y+h),(0,255,0),3)
     cv2.imshow("Back ground",delta_frame)
     cv2.imshow("Capture",gray) #this will show whatever is there  in front of your camera and as we are iterating this in while loop that will form a Video
     cv2.imshow("Threshold",thresh_frame)
     cv2.imshow  ("Color frame",frame)
    # print(gray)
     key = cv2.waitKey(1) #wait key is important function, the integer inside it corresponds to the number of second a frame would be visible ...and variable key would be used to break the loop
     if key == ord('q'):
         break
video.release() #this will close the camera
cv2.destroyAllWindows() #this will destroy all the window file appeared
