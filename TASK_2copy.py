import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images and preprocess
def load_image(image_path, target_size=(1024, 1024)):  
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Cannot open image at {image_path}. Check the path!")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)  
    image = np.array(image, dtype=np.float32) / 255.0  
    return image

# Load pre-trained High-Resolution Style Transfer model
style_transfer_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Function to apply style transfer
def apply_style_transfer(content_path, style_path):
    content_image = load_image(content_path)
    style_image = load_image(style_path, target_size=(300, 300))

    # Convert images to tensors with explicit dtype=tf.float32
    content_tensor = tf.convert_to_tensor(content_image, dtype=tf.float32)[tf.newaxis, ...]
    style_tensor = tf.convert_to_tensor(style_image, dtype=tf.float32)[tf.newaxis, ...]

    # ✅ Ensure correct dtype
    content_tensor = tf.image.convert_image_dtype(content_tensor, dtype=tf.float32)
    style_tensor = tf.image.convert_image_dtype(style_tensor, dtype=tf.float32)

    # Apply style transfer
    stylized_image = style_transfer_model(content_tensor, style_tensor)[0]

    # Convert tensor to image
    output_image = np.clip(stylized_image.numpy() * 255, 0, 255).astype(np.uint8)

    # ✅ Apply Post-Processing for Better Quality
    output_image = cv2.bilateralFilter(output_image, 9, 75, 75)  

    # Show image
    plt.figure(figsize=(10, 10))  
    plt.imshow(output_image)
    plt.axis('off')
    plt.show()

    return output_image

# Example Usage
content_image_path = r"C:\Users\Sachin m\Pictures\sklogo.jpg" 
style_image_path = r"C:\Users\Sachin m\Pictures\Camera Roll\Screenshots\Screenshot 2023-08-10 172707.png"  

# Apply Fast High-Quality Style Transfer
styled_image = apply_style_transfer(content_image_path, style_image_path)