import cv2
import numpy as np

def size_extract(file_name):
    width,length=0
    lower_green = np.array([20, 50, 50])
    upper_green = np.array([85, 255, 255])
    lower_yellow = np.array([18, 100, 100])
    upper_yellow = np.array([35, 160, 255])
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([20, 255, 255])
    
    if file_name.endswith(".jpg") or file_name.endswith(".png"):  # Process only .jpg and .png files
            # Read the input image
            image = cv2.imread(file_name)
            if image is None:
                print(f"Error: Unable to read the image {file_name}.")

            image = cv2.resize(image, (640, 480), cv2.INTER_AREA)

            smoothed_image = cv2.GaussianBlur(image, (15, 15), 0)

             # Convert the image from BGR to HSV color space
            hsv_image = cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2HSV)


            # Create masks to isolate the objects
            mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
            mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
            mask_orange = cv2.inRange(hsv_image, lower_orange, upper_orange)
            
            # Combine masks
            mask = cv2.bitwise_or(mask_green, mask_yellow)
            mask = cv2.bitwise_or(mask, mask_orange)

             # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                n=0
                if cv2.contourArea(contour) > 800:  # Filter out small contours
                    area = cv2.contourArea(contour)
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Write the dimensions to the file
                    if w > h:
                        n=h
                        h=w
                        w=n
                    width=w
                    length=h

    return width,length


