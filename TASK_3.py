import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg19
from tensorflow.keras.models import Model

def load_and_process_image(image_path, target_size=(400, 400)):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Cannot open image at {image_path}. Check the path!")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.vgg19.preprocess_input(image)
    return tf.convert_to_tensor(image, dtype=tf.float32)

def deprocess_image(image_tensor):
    image = image_tensor.numpy()
    image = image.reshape((image.shape[1], image.shape[2], 3))
    image[:, :, 0] += 103.939
    image[:, :, 1] += 116.779
    image[:, :, 2] += 123.68
    image = image[:, :, ::-1]
    image = np.clip(image, 0, 255).astype('uint8')
    return image

def compute_loss(combination_image, content_image, style_image):
    model = vgg19.VGG19(weights='imagenet', include_top=False)
    model_outputs = [layer.output for layer in model.layers]
    feature_extractor = Model(inputs=model.input, outputs=model_outputs)
    
    features = feature_extractor(tf.concat([content_image, style_image, combination_image], axis=0))
    
    content_loss = tf.reduce_mean(tf.square(features[5][0] - features[5][2]))
    style_loss = tf.reduce_mean(tf.square(features[1][1] - features[1][2]))
    
    total_loss = content_loss + 1e-4 * style_loss
    return total_loss

def neural_style_transfer(content_path, style_path, iterations=500, lr=0.02):
    content_image = load_and_process_image(content_path)
    style_image = load_and_process_image(style_path)
    
    # Correctly initialize combination image
    combination_image = tf.Variable(tf.identity(content_image), trainable=True)

    optimizer = tf.optimizers.Adam(learning_rate=lr) 
    

    for i in range(iterations):
        with tf.GradientTape() as tape:
            loss = compute_loss(combination_image, content_image, style_image)
        grads = tape.gradient(loss, combination_image)
        optimizer.apply_gradients([(grads, combination_image)])
        
        if i % 100 == 0:
            print(f"Iteration {i}: Loss {loss.numpy()}")

    output_image = deprocess_image(combination_image)
    plt.imshow(output_image)
    plt.axis('off')
    plt.show()
    return output_image

# Example Usage
content_image_path = r"C:\Users\Sachin m\Pictures\New folder\MM-normal.jpg"  # Replace with your image
style_image_path = r"C:\Users\Sachin m\Pictures\Camera Roll\Screenshots\Screenshot 2023-08-10 172707.png"  # Replace with your style image
styled_image = neural_style_transfer(content_image_path, style_image_path)
