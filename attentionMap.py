# -------------------------------------------------#
#visualizing which parts of an input image contribute most to your model's predictions

def compute_grad_cam(model, image, layer_name, class_idx=None):
    """
    Compute Grad-CAM for the given model, image, and layer name.
    """
    grad_model = Model_keras(
        inputs=[model.input],
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model([image])
        if class_idx is None:
            class_idx = tf.argmax(predictions[0])  # Get the predicted class index
        class_channel = predictions[:, class_idx]

    # Compute the gradient of the top predicted class with respect to the convolutional output
    grads = tape.gradient(class_channel, conv_output)
    # Compute the mean intensity of the gradient
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel by "how important" it is with regard to the class
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_output, axis=-1)


    # Normalize the heatmap to a range of 0 to 1 for visualization
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    return heatmap

def overlay_heatmap(heatmap, image, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """
    Overlay the heatmap on the image for visualization.
    """
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay

# Example usage with the thermal image branch
sample_image=X_test["natural"][0]
sample_image_batch ={"thermal":np.expand_dims(X_test["thermal"][0],axis=0),
"natural":np.expand_dims(X_test["natural"][0],axis=0),
"aerosol":np.expand_dims(X_test["aerosol"][0],axis=0),
"optical_thickness":np.expand_dims(X_test["optical_thickness"][0],axis=0),
"moisture":np.expand_dims(X_test["moisture"][0],axis=0),
"chlorophyll":np.expand_dims(X_test["chlorophyll"][0],axis=0)
}
layer_name = 'conv2d_2'  # Replace with the actual convolutional layer name from your model

heatmap = compute_grad_cam(model, sample_image_batch, layer_name)
overlay = overlay_heatmap(heatmap, (sample_image * 255).astype(np.uint8))  # Denormalize the image for visualization

# Plot the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow((sample_image * 255).astype(np.uint8))  # Denormalize the image
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Grad-CAM Overlay")
plt.imshow(overlay)
plt.axis('off')

plt.show()
plt.savefig("plots/AttnMap_FirstConvNatural.png")


# ---------------------------------------------------#end of attention maps
