import cv2
import numpy as np
from sklearn.cluster import KMeans

import tkinter as tk
from tkinter import Canvas, Frame
from PIL import Image, ImageTk, ImageFilter
import cv2
import numpy as np
from sklearn.cluster import KMeans

texture_map = r'./texture_map.png'
labeled_texture_map = r'./labeled_texture_map.png'

class ImageAppUsingML:
    def __init__(self, root, original_img_path):
        self.root = root
        self.root.title("Image App Using ML")

        # Load original image
        self.original_img = cv2.imread(original_img_path)
        self.original_img_rgb = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
        self.original_pil = Image.fromarray(self.original_img_rgb)

        # Create canvas for displaying images
        self.canvas = Canvas(root, width=self.original_pil.width * 2, height=self.original_pil.height)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a frame for the scale and buttons
        self.control_frame = Frame(root)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Create Scale for threshold adjustment
        self.threshold_scale = tk.Scale(self.control_frame, from_=5, to_=10, orient=tk.VERTICAL, label="Threshold")
        self.threshold_scale.pack()
        self.threshold_scale_1 = tk.Scale(self.control_frame, from_=0, to_=255, orient=tk.VERTICAL, label="Threshold_1")
        self.threshold_scale_1.pack()

        # Bind the scale to update the image
        self.threshold_scale.bind("<Motion>", self.update_image)
        self.threshold_scale_1.bind("<Motion>", self.update_image)

        # Initial display
        self.update_image()
    
    def convert_rgb_to_hsv(self, image):
        # Convert from RGB to HSV using OpenCV
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    def cluster_pixels(self, image, n_clusters=5):
        # Reshape the image to a 2D array of pixels
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # Create and fit the KMeans model
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)
        
        # Get the cluster centers (mean colors)
        cluster_centers = kmeans.cluster_centers_.astype(np.uint8)
        
        # Predict the cluster for each pixel
        labels = kmeans.predict(pixels)
        
        return labels, cluster_centers

    def apply_kmeans_to_image(self, labels, cluster_centers, image_shape):
        # Map each pixel to its cluster center color
        new_pixels = np.array([cluster_centers[label] for label in labels], dtype=np.uint8)
        
        # Reshape the result back to the original image shape
        new_image = new_pixels.reshape(image_shape)
        
        return new_image

    def convert_hsv_to_rgb(self, image):
        # Convert from HSV to RGB using OpenCV
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    def update_image(self, event=None):
        # Get the threshold value from the scale
        threshold = self.threshold_scale.get()
        threshold_1 = self.threshold_scale_1.get()

        # Convert RGB to HSV
        hsv_image = self.convert_rgb_to_hsv(self.original_img_rgb)
        
        # Cluster the pixels in the HSV color space
        labels, cluster_centers = self.cluster_pixels(hsv_image, n_clusters=10)  # Adjust number of clusters as needed
        
        # Apply the K-Means result to the image
        clustered_hsv_image = self.apply_kmeans_to_image(labels, cluster_centers, hsv_image.shape)
        
        # Convert the clustered HSV image back to RGB
        clustered_rgb_image = self.convert_hsv_to_rgb(clustered_hsv_image)
        
        # Convert RGB back to BGR for OpenCV
        clustered_bgr_image = cv2.cvtColor(clustered_rgb_image, cv2.COLOR_RGB2BGR)

        thresh_img = clustered_rgb_image

        cv2.imwrite(labeled_texture_map, thresh_img)
        
        # Convert to PIL format for display
        thresh_pil = Image.fromarray(thresh_img)

        # Convert images to PhotoImage format
        original_tk = ImageTk.PhotoImage(self.original_pil)
        thresh_tk = ImageTk.PhotoImage(thresh_pil)

        # Clear the canvas and display both images
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=original_tk)
        self.canvas.create_image(self.original_pil.width, 0, anchor=tk.NW, image=thresh_tk)

        # Update canvas size and scroll region
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

        # Keep references to avoid garbage collection
        self.original_tk = original_tk
        self.thresh_tk = thresh_tk

class ImageApp:
    def __init__(self, root, original_img_path):
        self.root = root
        self.root.title("Image Threshold Adjuster")

        # Load original image
        self.original_img = cv2.imread(original_img_path)
        self.original_img_rgb = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
        self.original_pil = Image.fromarray(self.original_img_rgb)

        # Create canvas for displaying images
        self.canvas = Canvas(root, width=self.original_pil.width * 2, height=self.original_pil.height)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a frame for the scale and buttons
        self.control_frame = Frame(root)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Create Scale for threshold adjustment
        self.threshold_scale = tk.Scale(self.control_frame, from_=0, to_=255, orient=tk.VERTICAL, label="Threshold")
        self.threshold_scale.pack()
        self.threshold_scale_1 = tk.Scale(self.control_frame, from_=0, to_=255, orient=tk.VERTICAL, label="Threshold_1")
        self.threshold_scale_1.pack()

        # Bind the scale to update the image
        self.threshold_scale.bind("<Motion>", self.update_image)
        self.threshold_scale_1.bind("<Motion>", self.update_image)

        # Initial display
        self.update_image()

    def identify_islands(self, uv_img: cv2.typing.MatLike, threshold: int):
        # Load the texture map
        gray = cv2.cvtColor(uv_img, cv2.COLOR_BGR2GRAY)
        
        # Threshold the image to binary (black and white)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours which represent different islands
        contours, _ = cv2.findContours(uv_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create an empty image to draw contours
        output_img = np.zeros_like(uv_img)
        
        # Loop through the contours and draw them
        for i, contour in enumerate(contours):
            # Generate a color for each island (for visualization purposes)
            color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            
            # Draw the contour on the output image
            cv2.drawContours(output_img, [contour], -1, color, thickness=cv2.FILLED)
        return output_img

    def fill_contours(self, uv_img, threshold, min_contour_area=100):
        # Convert to grayscale
        gray_img = cv2.cvtColor(uv_img, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred_img, threshold, 255)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty image to draw filled contours
        filled_contours_img = np.zeros_like(uv_img)

        # Iterate through contours and filter by area
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_contour_area:
                cv2.drawContours(filled_contours_img, [contour], -1, (0, 255, 0), thickness=cv2.FILLED)
        return filled_contours_img

    def detect_edges_rgb(self, uv_img: cv2.typing.MatLike):
        uv_img = cv2.GaussianBlur(uv_img, (5, 5), 0)
        # Split the image into its B, G, R channels
        b_channel, g_channel, r_channel = cv2.split(uv_img)

        # Apply Canny edge detection on each channel
        edges_b = cv2.Canny(b_channel, 50, 150)
        edges_g = cv2.Canny(g_channel, 50, 150)
        edges_r = cv2.Canny(r_channel, 50, 150)

        # Combine the edges from each channel
        edges_combined = np.maximum(edges_b, np.maximum(edges_g, edges_r))

        # Convert edges to a 3-channel image
        edges_colored = cv2.cvtColor(edges_combined, cv2.COLOR_GRAY2BGR)

        # Overlay edges on the original RGB image
        overlay_img = cv2.addWeighted(uv_img, 0.8, edges_colored, 1.0, 0)

        return overlay_img

    def adjust_saturation_value(self, uv_img, s_threshold, v_threshold):
        
        # Convert the image from RGB to HSV
        hsv_image = cv2.cvtColor(uv_img, cv2.COLOR_BGR2HSV)
        
        # Extract HSV channels
        h, s, v = cv2.split(hsv_image)
        
        # Set saturation and value to 100%
        s[:] = s_threshold  # Saturation max value in OpenCV
        v[:] = v_threshold  # Value max value in OpenCV
        
        # Merge the channels back
        hsv_image = cv2.merge([h, s, v])
        
        # Convert the HSV image back to RGB
        rgb_adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        
        return rgb_adjusted_image
    
    def update_image(self, event=None):
        # Get the threshold value from the scale
        threshold = self.threshold_scale.get()
        threshold_1 = self.threshold_scale_1.get()

        # thresh_img = self.identify_islands(self.original_img, threshold)
        # thresh_img = self.detect_edges_rgb(self.original_img)
        # thresh_img = self.fill_contours(self.original_img, threshold, 0)
        thresh_img = self.adjust_saturation_value(self.original_img, threshold, threshold_1)

        # Convert to PIL format for display
        thresh_pil = Image.fromarray(thresh_img)

        # Convert images to PhotoImage format
        original_tk = ImageTk.PhotoImage(self.original_pil)
        thresh_tk = ImageTk.PhotoImage(thresh_pil)

        # Clear the canvas and display both images
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=original_tk)
        self.canvas.create_image(self.original_pil.width, 0, anchor=tk.NW, image=thresh_tk)

        # Update canvas size and scroll region
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

        # Keep references to avoid garbage collection
        self.original_tk = original_tk
        self.thresh_tk = thresh_tk

if __name__ == "__main__":
    root = tk.Tk()
    # app = ImageApp(root, texture_map)
    app = ImageAppUsingML(root, texture_map)
    root.mainloop()
