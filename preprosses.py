import cv2
import numpy as np
import time

def compute_noise(image):
    """
    Compute the noise level of an image using standard deviation.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        float: Noise level (standard deviation of pixel intensities).
    """
    start_time = time.time()

    if len(image.shape) == 3:  # Convert to grayscale if needed
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    noise_level = np.std(image)
    end_time = time.time()
    time_taken = end_time - start_time
    fps = 1 / time_taken if time_taken > 0 else float('inf')
    
    print(f"calc noise took {time_taken:.4f} seconds, FPS: {fps:.2f}")

    return noise_level


def apply_nlm_denoising(frame, h=10, hForColorComponents=10, templateWindowSize=7, searchWindowSize=21):
    """
    Apply Non-Local Means (NLM) denoising to a single frame.

    Args:
        frame (numpy.ndarray): Input frame (image).
        h (int): Filter strength for luminance (grayscale) noise reduction.
        hForColorComponents (int): Same as h but for color images.
        templateWindowSize (int): Size of the template patch that is used to compute weights.
        searchWindowSize (int): Size of the window used to search for patches.

    Returns:
        numpy.ndarray: Denoised image.
    """
    start_time = time.time()

    denoised_image = cv2.fastNlMeansDenoisingColored(
        frame, None, h, hForColorComponents, templateWindowSize, searchWindowSize
    )

    end_time = time.time()
    time_taken = end_time - start_time
    fps = 1 / time_taken if time_taken > 0 else float('inf')
    
    print(f"Denoising took {time_taken:.4f} seconds, FPS: {fps:.2f}")

    return denoised_image
