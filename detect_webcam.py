import cv2
import json
import datetime
import time
import imutils

with open('config.json') as f:
    config = json.load(f)

if not config:
    raise RuntimeError('Could not parse config.json.')

if not config['detect_motion'] and not config['detect_birds']:
    raise RuntimeError('No detection methods enabled. Please enable motion and/or bird detection.')

cap = cv2.VideoCapture(config['webcam_id'])

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

fps = config['framerate']
frame_time = int(1000 / fps)

bird_class = cv2.CascadeClassifier('cascade.xml')

while True:
    ret, img_orig = cap.read()
    if not ret: break
    img = img_orig.copy()

    scaleX = config['process_width'] / img.shape[1]
    scaleY = config['process_height'] / img.shape[0]
    scale = min(scaleX, scaleY)
    if scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    x1 = config['roi_X']
    x2 = config['roi_X2']
    y1 = config['roi_Y']
    y2 = config['roi_Y2']
    img_gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

    motion_detected = False
    bird_detected = False

    if config['detect_motion']:
        mask = cv2.GaussianBlur(img_gray, (13, 13), 0) # smooth the image
        mask = fgbg.apply(mask) # remove background
        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < config['motion_min_area']:
                continue
            
            motion_detected = True

            if config['show_video'] or config['output_debug']:
                # compute the bounding box for the contour, draw it on the frame
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            else:
                # Can stop processing as soon as motion is detected if we aren't debugging
                break
        if motion_detected: print('Detected motion')
        if config['show_mask']:
            cv2.imshow('mask', mask)

    if config['detect_birds'] and (motion_detected or not config['detect_motion']):
        # only run bird detection if active
        # if motion detection active then require motion first

        # Use minSize because we're not 
        # bothering with extra-small 
        # dots that would look like birds
        minsize = config['bird_min_size']
        maxsize = config['bird_max_size']
        found = bird_class.detectMultiScale(img_gray, 
                    minSize=(minsize, minsize),
                    maxSize=(maxsize, maxsize)) 
        # Don't do anything if there's no bird
        amount_found = len(found) 
        if amount_found > 0:
            bird_detected = True
            print(f'Found {amount_found} birds')
            if config['show_video'] or config['output_debug']:
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

    should_save = (motion_detected or not config['detect_motion']) and (bird_detected or not config['detect_birds'])
    if should_save:
        folder = config['output_folder']
        if folder:
            curTime = datetime.datetime.now().replace(microsecond=0).isoformat()
            filename = curTime.replace(':', '.')
            extension = '.jpg'
            cv2.imwrite(f'{folder}/{filename}{extension}', img_orig)

            if config['output_debug']:
                cv2.imwrite(f'{folder}/{filename}_debug{extension}', img)
                cv2.imwrite(f'{folder}/{filename}_mask{extension}', mask)


    if config['show_video']:
        cv2.imshow('Video Feed', img)
    if config['show_video'] or config['show_mask']:
        c = cv2.waitKey(frame_time)
        if c == 27: # Escape key
            break
    else:
        time.sleep(frame_time/1000)

cap.release()
cv2.destroyAllWindows()