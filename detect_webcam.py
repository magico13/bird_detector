import cv2

cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

fps = 5
frame_time = int(1000 /fps)

bird_class = cv2.CascadeClassifier('cascade.xml')

while True:
    ret, img = cap.read()
    if not ret: break

    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    x1 = 0
    x2 = img.shape[1]
    # y1 = 100
    # y2 = 350
    y1 = 0
    y2 = img.shape[0]
    img_gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(img_gray, (13, 13), 0)
    fgmask = fgbg.apply(blurred)

    # Use minSize because we're not 
    # bothering with extra-small 
    # dots that would look like birds
    minsize = 25
    maxsize = 250
    found = bird_class.detectMultiScale(img_gray, 
                minSize=(minsize, minsize),
                maxSize=(maxsize, maxsize)) 
    # Don't do anything if there's 
    # no bird
    amount_found = len(found) 
    
    if amount_found > 0:
        print(f'Found {amount_found} birds')
        # There may be more than one 
        # bird in the image 
        for (x, y, w, h) in found: 
            # We draw a green rectangle around 
            # every recognized bird 

            #adjust for roi
            x += x1
            y += y1

            cv2.rectangle(img, (x, y), 
                        (x + h, y + w), 
                        (0, 255, 0), 5)

    cv2.imshow('Video Feed', img)
    cv2.imshow('mask',fgmask)
    c = cv2.waitKey(frame_time)
    if c == 27: # Escape key
        break

cap.release()
cv2.destroyAllWindows()